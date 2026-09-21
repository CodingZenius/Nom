"""Paths, defaults, config.json + .env handling.

Everything mutable lives under NOMAD_HOME (default: ./.nomad next to the code), and
arch-specific binaries are stored per-architecture so a CPU change (x86 <-> arm) never
clobbers the other set.
"""
from __future__ import annotations
import copy
import json
import os
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME = Path(os.environ.get("NOMAD_HOME", ROOT / ".nomad")).expanduser().resolve()
_m = platform.machine().lower()
ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(_m, _m)

WORKSPACE = Path(os.environ.get("NOMAD_WORKSPACE", HOME / "workspace")).expanduser().resolve()
MD_DIR = Path(os.environ.get("NOMAD_MD_DIR", ROOT)).expanduser().resolve()
MEMORY_MD = MD_DIR / "Memory.md"
PERSONALITY_MD = MD_DIR / "Personality.md"

MODELS_DIR = HOME / "ollama_models"          # model weights are CPU-independent -> shared
OLLAMA_DIR = HOME / "ollama" / ARCH          # binary is arch-specific
BROWSERS_DIR = HOME / "browsers" / ARCH      # chromium is arch-specific
SHOTS_DIR = HOME / "screenshots"
LOGS_DIR = HOME / "logs"
CONFIG_PATH = HOME / "config.json"
STATE_PATH = HOME / "state.json"
ENV_PATH = HOME / ".env"
SQLITE_PATH = HOME / "nomad.sqlite"

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")

DEFAULTS: dict = {
    "version": 1,
    "model": {"provider": "ollama", "name": "qwen2.5:3b", "num_ctx": None,
              "num_thread": None, "temperature": 0.2, "keep_alive": "30m"},
    # {"provider": "nim"|"gemini", "name": "...", "num_ctx": 32768, "role": "planner+fallback"}
    "aux": None,
    "agent": {"max_steps": 10, "task_retries": 2, "human_browsing": True, "unsafe_paths": False},
    # {"endpoint_url": "...", "bucket": "...", "region": "auto", "prefix": "nomad/"}  (keys via env)
    "s3": None,
}


def ensure_dirs() -> None:
    for d in (HOME, MODELS_DIR, OLLAMA_DIR, BROWSERS_DIR, SHOTS_DIR, LOGS_DIR, WORKSPACE):
        d.mkdir(parents=True, exist_ok=True)


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config() -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg = deep_merge(cfg, json.loads(CONFIG_PATH.read_text()))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    HOME.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def load_env() -> None:
    """Load .env files without overriding real environment variables."""
    for p in (ROOT / ".env", ENV_PATH):
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def update_env(pairs: dict) -> None:
    """Persist secrets to NOMAD_HOME/.env (chmod 600) and the live environment."""
    HOME.mkdir(parents=True, exist_ok=True)
    cur = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                cur[k.strip()] = v.strip()
    for k, v in pairs.items():
        if v:
            cur[k] = v
            os.environ[k] = v
    ENV_PATH.write_text("".join(f"{k}={v}\n" for k, v in cur.items()))
    try:
        os.chmod(ENV_PATH, 0o600)
    except OSError:
        pass


def state_get(key: str, default=None):
    try:
        return json.loads(STATE_PATH.read_text()).get(key, default)
    except Exception:
        return default


def state_set(key: str, value) -> None:
    try:
        d = json.loads(STATE_PATH.read_text())
    except Exception:
        d = {}
    d[key] = value
    HOME.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(d, indent=2))
