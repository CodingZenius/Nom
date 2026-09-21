"""Helpers: tolerant JSON extraction from small-model output, truncation, token estimates."""
from __future__ import annotations
import ast
import datetime as dt
import json
import re


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def trunc(s, n: int = 1500, tail: int = 0) -> str:
    s = "" if s is None else str(s)
    if len(s) <= n:
        return s
    if tail:
        head = max(0, n - tail - 24)
        return s[:head] + f"\n...[{len(s) - n} chars cut]...\n" + s[-tail:]
    return s[:n] + f"...[+{len(s) - n} chars]"


def est_tokens(s: str) -> int:
    """Conservative estimate (code/JSON tokenise worse than prose)."""
    return len(s) // 3 + 1


def slug(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s or "").strip("-").lower()[:n] or "x"


def _loads(s: str):
    for fix in (lambda x: x, lambda x: re.sub(r",\s*([}\]])", r"\1", x)):
        try:
            return json.loads(fix(s), strict=False)  # strict=False: raw newlines inside strings
        except Exception:
            pass
    try:  # python-style dicts: {'a': True}
        v = ast.literal_eval(s)
        if isinstance(v, (dict, list)):
            return v
    except Exception:
        pass
    return None


def extract_json(text: str):
    """Return the first JSON object/array found in `text` (dict/list) or None."""
    if not text:
        return None
    t = text.strip()
    t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    v = _loads(t)
    if isinstance(v, (dict, list)):
        return v
    tries = 0
    for start, ch in enumerate(t):
        if ch not in "{[":
            continue
        tries += 1
        if tries > 40:
            break
        close = "}" if ch == "{" else "]"
        depth, in_s, esc = 0, False, False
        for i in range(start, len(t)):
            c = t[i]
            if in_s:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_s = False
                continue
            if c == '"':
                in_s = True
            elif c == ch:
                depth += 1
            elif c == close:
                depth -= 1
                if depth == 0:
                    v = _loads(t[start:i + 1])
                    if isinstance(v, (dict, list)):
                        return v
                    break
    return None
