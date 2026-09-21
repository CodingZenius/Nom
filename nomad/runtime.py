"""Wire everything together and adapt to whatever machine we just woke up on."""
from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, field

from . import bootstrap, hardware, onboarding
from . import config as C
from .agent import Agent
from .browser import Browser
from .db import Store
from .llm import build_llms, ollama_model_ctx
from .memory import Memory
from .planner import Planner
from .shell import Workspace
from .storage import Blob
from .tasks import TaskManager
from .tools import Toolbox
from .ui import say
from .util import now_iso


@dataclass
class App:
    cfg: dict
    hw: dict
    store: Store
    memory: Memory
    llm: object
    aux: object
    blob: Blob
    browser: object
    tools: Toolbox
    tm: TaskManager
    agent: Agent
    planner: Planner
    report: dict = field(default_factory=dict)

    def close(self):
        try:
            if self.browser:
                self.browser.close()
        finally:
            self.store.close()

    def sync(self):
        self.memory.push()
        self.store.kv_set("config", json.dumps(self.cfg))


def _restore_config_from_db() -> None:
    """Fresh machine (e.g. new Colab VM) with DATABASE_URL set: pull the saved config."""
    url = os.environ.get("DATABASE_URL")
    if not url or C.CONFIG_PATH.exists():
        return
    try:
        tmp = Store(url)
        v = None if tmp.error else tmp.kv_get("config")
        tmp.close()
        if v:
            C.HOME.mkdir(parents=True, exist_ok=True)
            C.CONFIG_PATH.write_text(v)
            say("info", "config restored from database")
    except Exception as e:
        say("warn", f"could not restore config: {e}")


def _track_hardware(store: Store, hw: dict) -> bool:
    """Compare against the last machine seen; log and report if the CPU/RAM/arch changed."""
    prev = None
    try:
        raw = store.kv_get("hardware")
        prev = json.loads(raw) if raw else C.state_get("hardware")
    except Exception:
        prev = C.state_get("hardware")
    changed = bool(prev and prev.get("fingerprint") != hw["fingerprint"])
    if changed:
        say("warn", f"hardware changed since last run:\n   was: {prev.get('cpu_model')} ({prev.get('arch')}, "
                    f"{prev.get('eff_cores')} cores, {prev.get('ram_total_gb')} GB)\n   now: {hardware.summary(hw)}\n"
                    "   -> re-verifying binaries and re-tuning threads/context")
        store.log("hardware_changed", {"before": prev, "after": hw})
    slim = {k: hw[k] for k in ("fingerprint", "arch", "cpu_model", "eff_cores", "ram_total_gb", "gpu")} | {"ts": now_iso()}
    C.state_set("hardware", slim)
    try:
        store.kv_set("hardware", json.dumps(slim))
    except Exception:
        pass
    return changed


def create(skip_boot: bool = False, interactive: bool | None = None) -> App:
    C.ensure_dirs()
    C.load_env()
    if interactive is None:
        interactive = sys.stdin.isatty()
    _restore_config_from_db()
    if not C.CONFIG_PATH.exists():
        say("info", "first run: starting onboarding" + ("" if interactive else " (non-interactive, using env/defaults)"))
        onboarding.run(interactive=interactive)
    cfg = C.load_config()

    store = Store(os.environ.get("DATABASE_URL"))
    hw = hardware.detect()
    say("info", "machine: " + hardware.summary(hw))
    _track_hardware(store, hw)

    report = {} if skip_boot else bootstrap.boot(cfg)

    max_ctx = ollama_model_ctx(cfg["model"]["name"]) if report.get("model_ready") else None
    rec = hardware.recommend(hw, cfg["model"]["name"], max_ctx)
    llm, aux = build_llms(cfg, rec)
    if report.get("errors") and cfg["model"]["provider"] == "ollama" and not report.get("model_ready"):
        if aux:
            say("warn", "local model unavailable - running on the auxiliary model until it recovers")
            llm = aux
        else:
            raise SystemExit("Local model unavailable and no auxiliary model configured:\n  " + "\n  ".join(report["errors"]))
    if llm.provider == "ollama":
        def _revive():
            b = bootstrap.find_ollama()
            if b:
                bootstrap.start_ollama(b)
        llm.recover = _revive
    say("ok", f"main={llm.name} ctx={llm.num_ctx} threads={llm.num_thread}" + (f" | aux={aux.provider}:{aux.name}" if aux else ""))

    blob = Blob(cfg.get("s3"))
    memory = Memory(store, blob, llm_getter=lambda: llm)
    memory.restore()
    human = bool(cfg["agent"].get("human_browsing", True))
    browser = Browser(exec_path=report.get("chromium_exec"), human=human) if (skip_boot or report.get("chromium")) else None
    ws = Workspace(C.WORKSPACE, unsafe=bool(cfg["agent"].get("unsafe_paths")), llm_getter=lambda: llm)
    tools = Toolbox(ws, browser, memory, blob)
    tm = TaskManager(store)
    agent = Agent(tools, memory, cfg, store)
    planner = Planner(llm, aux, tm, agent, store, memory, cfg)
    app = App(cfg, hw, store, memory, llm, aux, blob, browser, tools, tm, agent, planner, report)
    try:
        store.kv_set("config", json.dumps(cfg))
    except Exception:
        pass
    return app
