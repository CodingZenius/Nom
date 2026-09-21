"""python run.py <command>

  onboard [--yes]     guided setup (models, context window, aux model, Neon, S3)
  boot                download/verify Ollama, model, Chromium; report hardware changes
  doctor              read-only health check (downloads nothing)
  chat                interactive session (each line is a goal)
  run "goal"          run one goal and exit (works in notebooks / CI)
  resume              continue the last unfinished goal
  shot URL [--mode human|headless] [--full]
  sync                push Memory.md / Personality.md / config to the database
"""
from __future__ import annotations
import argparse
import os
import sys

from . import bootstrap, hardware, onboarding
from . import config as C
from .ui import say


def cmd_doctor() -> int:
    C.load_env()
    hw = hardware.detect()
    say("info", "machine: " + hardware.summary(hw))
    print(f"  fingerprint {hw['fingerprint']}  flags {','.join(hw['cpu_flags']) or '-'}  root={hw['root']}")
    cfg = C.load_config()
    print(f"  home        {C.HOME}\n  workspace   {C.WORKSPACE}\n  config      {'present' if C.CONFIG_PATH.exists() else 'MISSING (run onboard)'}")
    ob = bootstrap.find_ollama()
    print(f"  ollama      {ob or 'NOT INSTALLED'}   server={'up' if bootstrap.ollama_up() else 'down'}")
    name = cfg["model"]["name"]
    print(f"  model       {name}: {'present' if bootstrap.has_model(name) else 'missing'}")
    ok, err = bootstrap.pw_launch_ok()
    print(f"  chromium    {'ok (playwright)' if ok else 'not ready: ' + err}")
    print(f"  aux model   {cfg['aux']['provider'] + ':' + cfg['aux']['name'] if cfg.get('aux') else 'none'}")
    url = os.environ.get("DATABASE_URL")
    if url:
        from .db import Store
        s = Store(url)
        print(f"  database    {'Neon OK' if not s.error else 'Neon FAILED: ' + s.error}")
        s.close()
    else:
        print("  database    none (local SQLite)")
    print(f"  s3          {cfg['s3']['bucket'] if cfg.get('s3') else 'off'}")
    return 0


def _chat(app) -> None:
    print("\nNomad ready. Type a goal, or /help.")
    un = app.store.goal_unfinished()
    if un:
        say("warn", f"unfinished goal: {un['goal'][:80]!r} ({app.tm.progress(un['id'])}) - type /resume to continue")
    while True:
        try:
            line = input("\nnomad> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if not line.startswith("/"):
            app.planner.run(line)
            continue
        cmd, _, arg = line[1:].partition(" ")
        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "help":
            print("/tasks /resume /memory /remember <fact> /shot <url> [human] /status /sync /quit")
        elif cmd == "tasks":
            g = app.store.goal_unfinished() or (app.store.goals_recent(1) or [None])[0]
            print(app.tm.render(g["id"]) if g else "(no goals yet)")
        elif cmd == "resume":
            g = app.store.goal_unfinished()
            print("nothing to resume" if not g else app.planner.run(goal_id=g["id"])["summary"])
        elif cmd == "memory":
            print(app.memory.memory_text())
        elif cmd == "remember":
            print(app.memory.remember(arg))
        elif cmd == "shot":
            parts = arg.split()
            if not parts:
                print("usage: /shot <url> [human]")
                continue
            app.tools.call("web_open", {"url": parts[0]})
            print(app.tools.call("web_shot", {"mode": "human" if "human" in parts[1:] else "headless"}))
        elif cmd == "status":
            print(f"db={app.store.dialect} main={app.llm} aux={app.aux} s3={'on' if app.blob.enabled else 'off'}")
        elif cmd == "sync":
            app.sync()
            print("synced")
        else:
            print("unknown command; /help")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="nomad", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("onboard"); p.add_argument("--yes", action="store_true", help="non-interactive (env vars)")
    sub.add_parser("boot")
    sub.add_parser("doctor")
    sub.add_parser("chat")
    p = sub.add_parser("run"); p.add_argument("goal", nargs="+")
    sub.add_parser("resume")
    p = sub.add_parser("shot"); p.add_argument("url"); p.add_argument("--mode", default="headless"); p.add_argument("--full", action="store_true")
    sub.add_parser("sync")
    a = ap.parse_args(argv)
    cmd = a.cmd or "chat"

    C.load_env()
    if cmd == "doctor":
        return cmd_doctor()
    if cmd == "onboard":
        onboarding.run(interactive=not a.yes and sys.stdin.isatty())
        return 0

    from . import runtime
    app = runtime.create(interactive=sys.stdin.isatty())
    try:
        if cmd == "boot":
            say("ok", "boot complete" + (f" with issues: {app.report.get('errors')}" if app.report.get("errors") else ""))
        elif cmd == "chat":
            _chat(app)
        elif cmd == "run":
            r = app.planner.run(" ".join(a.goal))
            app.sync()
            return 0 if r["status"] == "done" else 1
        elif cmd == "resume":
            g = app.store.goal_unfinished()
            if not g:
                say("info", "nothing to resume")
                return 0
            r = app.planner.run(goal_id=g["id"])
            app.sync()
            return 0 if r["status"] == "done" else 1
        elif cmd == "shot":
            app.tools.call("web_open", {"url": a.url})
            print(app.tools.call("web_shot", {"mode": a.mode, "full": a.full}))
        elif cmd == "sync":
            app.sync()
            say("ok", "synced")
    finally:
        try:
            app.sync()
        except Exception:
            pass
        app.close()
    return 0
