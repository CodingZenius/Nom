"""Workspace tools: navigate the file system, write/edit/revise code, run shell commands.

File tools are confined to the workspace root (unless agent.unsafe_paths is true).
The container itself is the real sandbox; `run` is only guarded by a small denylist.
"""
from __future__ import annotations
import difflib
import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

from .util import trunc

DENY = re.compile(
    r"(\brm\s+-[a-zA-Z]*[rf][a-zA-Z]*\s+(/|~|\$HOME)(\s|$)|\bmkfs\b|:\(\)\s*\{|\bdd\s+if=.*\bof=/dev/|"
    r">\s*/dev/sd|\b(shutdown|reboot|halt|poweroff)\b)")
IGNORE = {".git", "__pycache__", "node_modules", ".venv", ".nomad", ".mypy_cache"}


class Workspace:
    def __init__(self, root: Path, unsafe: bool = False, llm_getter=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.unsafe = unsafe
        self.llm_getter = llm_getter

    # ------------------------------------------------------------ paths
    def resolve(self, p) -> Path:
        q = Path(str(p)).expanduser()
        full = (q if q.is_absolute() else self.root / q).resolve()
        if not self.unsafe:
            try:
                full.relative_to(self.root)
            except ValueError:
                raise PermissionError(f"path outside workspace ({self.root}): {full}")
        return full

    # ------------------------------------------------------------ shell
    def run(self, cmd: str, timeout: int = 120) -> str:
        if DENY.search(cmd):
            return "ERROR: command blocked by safety filter"
        timeout = max(1, min(int(timeout), 900))
        sh = shutil.which("bash") or "sh"
        p = subprocess.Popen([sh, "-c", cmd], cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, errors="replace", start_new_session=True)
        try:
            out, _ = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except Exception:
                p.kill()
            out, _ = p.communicate()
            return trunc(out, 2500, tail=800) + f"\nERROR: timed out after {timeout}s (killed)"
        return f"exit={p.returncode}\n" + trunc(out, 3500, tail=1000)

    # ------------------------------------------------------------ navigate
    def ls(self, path: str = ".", depth: int = 1) -> str:
        base = self.resolve(path)
        if not base.exists():
            return f"ERROR: {path} does not exist"
        if base.is_file():
            return f"{base.name} (file, {base.stat().st_size} bytes)"
        depth = max(1, min(int(depth), 4))
        lines: list[str] = []

        def walk(d: Path, level: int):
            for e in sorted(d.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if e.name in IGNORE or len(lines) > 200:
                    continue
                lines.append("  " * level + e.name + ("/" if e.is_dir() else f"  ({e.stat().st_size}b)"))
                if e.is_dir() and level + 1 < depth:
                    walk(e, level + 1)

        walk(base, 0)
        rel = os.path.relpath(base, self.root)
        return f"{rel}/\n" + ("\n".join(lines) if lines else "(empty)")

    def read_file(self, path: str, start: int = 1, end: int = 200) -> str:
        p = self.resolve(path)
        if not p.is_file():
            return f"ERROR: {path} is not a file"
        try:
            lines = p.read_text().splitlines()
        except UnicodeDecodeError:
            return "ERROR: binary file"
        start = max(1, int(start))
        end = min(len(lines), int(end), start + 299)
        body = "\n".join(f"{i + 1:>4}| {lines[i]}" for i in range(start - 1, end))
        return f"{p.name} lines {start}-{end} of {len(lines)}\n" + trunc(body, 6000)

    def search(self, pattern: str, path: str = ".") -> str:
        base = self.resolve(path)
        try:
            rx = re.compile(pattern)
        except re.error:
            rx = re.compile(re.escape(pattern))
        hits: list[str] = []
        files = [base] if base.is_file() else (f for f in base.rglob("*") if f.is_file())
        for f in files:
            if any(part in IGNORE for part in f.parts) or f.stat().st_size > 1_000_000:
                continue
            try:
                for n, line in enumerate(f.read_text().splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{os.path.relpath(f, self.root)}:{n}: {line.strip()[:160]}")
                        if len(hits) >= 40:
                            return "\n".join(hits) + "\n...(40 hit limit)"
            except (UnicodeDecodeError, OSError):
                continue
        return "\n".join(hits) or "no matches"

    # ------------------------------------------------------------ write / edit / revise
    @staticmethod
    def _syntax(p: Path) -> str:
        if p.suffix == ".py":
            try:
                compile(p.read_text(), str(p), "exec")
            except SyntaxError as e:
                return f"\nWARNING: python syntax error line {e.lineno}: {e.msg}"
        return ""

    def write_file(self, path: str, content: str) -> str:
        p = self.resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content if isinstance(content, str) else str(content))
        return f"wrote {p.stat().st_size} bytes, {len(content.splitlines())} lines to {path}" + self._syntax(p)

    def edit_file(self, path: str, old: str, new: str) -> str:
        p = self.resolve(path)
        if not p.is_file():
            return f"ERROR: {path} is not a file"
        txt = p.read_text()
        n = txt.count(old)
        if not old:
            return "ERROR: 'old' must not be empty"
        if n == 0:
            return "ERROR: 'old' text not found. Use read_file to copy the exact text (whitespace matters)."
        if n > 1:
            return f"ERROR: 'old' matches {n} places; include more surrounding lines to make it unique."
        p.write_text(txt.replace(old, new, 1))
        return f"edited {path}" + self._syntax(p)

    def revise_file(self, path: str, instruction: str) -> str:
        """Have the LLM rewrite a small file per the instruction; verify syntax; show a diff."""
        llm = self.llm_getter() if self.llm_getter else None
        if llm is None:
            return "ERROR: no model available"
        p = self.resolve(path)
        if not p.is_file():
            return f"ERROR: {path} is not a file"
        old = p.read_text()
        if len(old) > 12000:
            return "ERROR: file too large for revise_file; use edit_file on specific lines instead"
        out = llm.chat([
            {"role": "system", "content": "You revise source files. Output ONLY the complete new file content. "
             "No markdown fences, no commentary. Change only what the instruction requires."},
            {"role": "user", "content": f"FILE {p.name}:\n{old}\n\nINSTRUCTION: {instruction}"}],
            max_tokens=min(4000, max(512, len(old) // 2)), temperature=0.1)
        out = re.sub(r"^```[a-zA-Z]*\n|\n```\s*$", "", out.strip()) + "\n"
        if len(out.strip()) < 10 or (len(old) > 400 and len(out) < len(old) * 0.3):
            return "ERROR: revision looked truncated; file NOT changed. Try edit_file instead."
        p.write_text(out)
        warn = self._syntax(p)
        if warn:
            p.write_text(old)
            return "ERROR: revision had a syntax problem, file restored." + warn
        diff = "".join(difflib.unified_diff(old.splitlines(1), out.splitlines(1), "before", "after", n=1))
        return f"revised {path}\n" + trunc(diff, 1800)
