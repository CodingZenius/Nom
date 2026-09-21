"""Memory.md (agent-written notes) and Personality.md (user-written persona/rules).

Files are the working copy; Neon (kv table) is the survival copy. On boot we reconcile:
  * local file unchanged since last sync  -> pull from DB if DB differs (fresh Colab/new machine)
  * local file edited by a human/agent     -> push to DB
  * never synced + both differ             -> Memory.md: union of lines; Personality.md: local wins
"""
from __future__ import annotations
import datetime as dt
import hashlib
import re

from . import config as C
from .ui import say

DEFAULT_PERSONALITY = """# Personality
Name: Nomad
Role: a careful, terse autonomous engineer-assistant running inside a container.

## Style
- Plain and direct. No filler. Report what you did and the result.
- Prefer small verified steps: change one thing, run it, read the output.

## Rules
- Never invent file contents, command output or URLs. Look first (ls, read_file, search).
- After writing or editing code, run it or a syntax check before calling done.
- Do only the CURRENT TASK. Keep your final summary under 3 sentences.
- Save durable facts (user preferences, project paths, environment quirks) with the remember tool. Never save secrets.
- If blocked twice on the same approach, change approach or call fail with the reason.
"""

DEFAULT_MEMORY = """# Memory
Durable notes kept by the agent. Edit freely.

## User
## Environment
## Facts
## Lessons
"""

MEM_MAX_CHARS = 6000
_FILES = (("Personality.md", "file:Personality.md", C.PERSONALITY_MD, DEFAULT_PERSONALITY),
          ("Memory.md", "file:Memory.md", C.MEMORY_MD, DEFAULT_MEMORY))


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def _merge_lines(a: str, b: str) -> str:
    have = {ln.strip() for ln in a.splitlines()}
    extra = [ln for ln in b.splitlines() if ln.strip() and not ln.startswith("#") and ln.strip() not in have]
    return a.rstrip("\n") + ("\n" + "\n".join(extra) if extra else "") + "\n"


class Memory:
    def __init__(self, store, blob=None, llm_getter=None):
        self.store, self.blob, self.llm_getter = store, blob, llm_getter

    # ------------------------------------------------------------ reconcile
    def restore(self) -> None:
        synced = C.state_get("synced", {}) or {}
        for name, key, path, default in _FILES:
            try:
                db_txt = self.store.kv_get(key)
                if not path.exists():
                    txt = db_txt if db_txt is not None else default
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(txt)
                    if db_txt is None:
                        self.store.kv_set(key, txt)
                    synced[name] = _h(txt)
                    continue
                local = path.read_text()
                if db_txt is None or db_txt == local:
                    if db_txt is None:
                        self.store.kv_set(key, local)
                    synced[name] = _h(local)
                    continue
                last = synced.get(name)
                untouched = (last == _h(local)) if last else (local == default)
                if untouched:                       # local is stale -> adopt DB copy
                    path.write_text(db_txt)
                    synced[name] = _h(db_txt)
                    say("info", f"{name}: restored from database")
                elif last:                          # local edited since last sync -> local wins
                    self.store.kv_set(key, local)
                    synced[name] = _h(local)
                elif name == "Memory.md":           # never synced, both differ -> merge
                    merged = _merge_lines(local, db_txt)
                    path.write_text(merged)
                    self.store.kv_set(key, merged)
                    synced[name] = _h(merged)
                    say("info", "Memory.md: merged local + database notes")
                else:
                    self.store.kv_set(key, local)
                    synced[name] = _h(local)
            except Exception as e:
                say("warn", f"{name} sync skipped: {e}")
        C.state_set("synced", synced)

    def push(self, name: str | None = None) -> None:
        synced = C.state_get("synced", {}) or {}
        for n, key, path, _ in _FILES:
            if name and n != name or not path.exists():
                continue
            txt = path.read_text()
            try:
                self.store.kv_set(key, txt)
                synced[n] = _h(txt)
            except Exception as e:
                say("warn", f"could not sync {n}: {e}")
            if self.blob and self.blob.enabled:
                try:
                    self.blob.put(f"state/{n}", txt.encode())
                except Exception as e:
                    say("warn", f"S3 snapshot of {n} failed: {e}")
        C.state_set("synced", synced)

    # ------------------------------------------------------------ read
    def personality(self) -> str:
        return C.PERSONALITY_MD.read_text() if C.PERSONALITY_MD.exists() else DEFAULT_PERSONALITY

    def memory_text(self) -> str:
        return C.MEMORY_MD.read_text() if C.MEMORY_MD.exists() else DEFAULT_MEMORY

    def context_block(self, max_chars: int) -> str:
        pers = self.personality().strip()[: max(600, max_chars // 3)]
        room = max(400, max_chars - len(pers))
        mem = self.memory_text().strip()
        if len(mem) > room:  # keep the newest material
            mem = mem[-room:]
            mem = mem[mem.find("\n") + 1:]
        return f"{pers}\n\n{mem}"

    # ------------------------------------------------------------ write
    def remember(self, fact: str, section: str = "Facts") -> str:
        fact = re.sub(r"\s+", " ", (fact or "").strip())
        if not fact:
            return "nothing to remember"
        if re.search(r"(api[_-]?key|secret|password|token)\s*[:=]", fact, re.I):
            return "refused: looks like a secret; not saved"
        section = re.sub(r"[^A-Za-z ]", "", section).strip().title() or "Facts"
        txt = self.memory_text()
        if fact.lower() in txt.lower():
            return "already known"
        lines = txt.rstrip("\n").split("\n")
        entry = f"- {dt.date.today().isoformat()}: {fact}"
        hdr = f"## {section}"
        if hdr not in lines:
            lines += ["", hdr]
        i = lines.index(hdr) + 1
        while i < len(lines) and not lines[i].startswith("## "):
            i += 1
        while i > 0 and not lines[i - 1].strip():
            i -= 1
        lines.insert(i, entry)
        out = "\n".join(lines) + "\n"
        if len(out) > MEM_MAX_CHARS:
            out = self._compact(out)
        C.MEMORY_MD.parent.mkdir(parents=True, exist_ok=True)
        C.MEMORY_MD.write_text(out)
        self.push("Memory.md")
        return "saved"

    def _compact(self, txt: str) -> str:
        llm = self.llm_getter() if self.llm_getter else None
        if llm:
            try:
                new = llm.chat([
                    {"role": "system", "content": "Rewrite the notes file shorter (under 3500 characters). Keep the exact "
                     "markdown headings, merge duplicates, drop stale or trivial entries, keep dates. Output only the file."},
                    {"role": "user", "content": txt}], max_tokens=1500, temperature=0.1).strip()
                if new.startswith("# Memory") and len(new) < len(txt) * 0.9:
                    return new + "\n"
            except Exception:
                pass
        # fallback: drop oldest bullet lines until it fits
        lines = txt.split("\n")
        while len("\n".join(lines)) > MEM_MAX_CHARS * 0.8:
            idx = next((i for i, ln in enumerate(lines) if ln.startswith("- ")), None)
            if idx is None:
                break
            del lines[idx]
        return "\n".join(lines) + "\n"
