"""Tiny console logger (works in terminals, Docker logs and Colab)."""
from __future__ import annotations
import sys

_COL = {"info": "36", "ok": "32", "warn": "33", "err": "31", "dl": "35",
        "setup": "35", "tool": "34", "think": "90", "plan": "36", "task": "1;36"}


def say(kind: str, msg: str) -> None:
    tag = f"[{kind}]"
    if sys.stdout.isatty():
        print(f"\033[{_COL.get(kind, '0')}m{tag}\033[0m {msg}", flush=True)
    else:
        print(tag, msg, flush=True)
