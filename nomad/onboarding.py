"""Onboarding: model choice, automatic context window, optional aux model, Neon, S3.

Interactive by default; with interactive=False (Colab / CI) everything comes from env vars:
  NOMAD_MODEL, NOMAD_NUM_CTX, NOMAD_AUX_PROVIDER (nim|gemini), NOMAD_AUX_MODEL, NOMAD_AUX_ROLE,
  NVIDIA_API_KEY / GEMINI_API_KEY (aux is inferred from whichever is set), DATABASE_URL,
  S3_ENDPOINT, S3_BUCKET, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY
"""
from __future__ import annotations
import getpass
import os

from . import config as C
from . import hardware
from .db import Store
from .llm import DEFAULT_AUX, LLM
from .storage import Blob
from .ui import say

MODELS = [
    ("qwen2.5:3b", "DEFAULT - ~2 GB, best size/quality for CPU"),
    ("qwen2.5:1.5b", "tiny - ~1 GB, for machines with <4 GB free RAM"),
    ("qwen2.5-coder:3b", "3B tuned for code"),
    ("qwen2.5:7b", "~4.7 GB, wants >=8 GB free RAM"),
    ("llama3.2:3b", "alternative 3B"),
]
ROLES = [("planner+fallback", "aux plans tasks and rescues failed steps"),
         ("fallback", "aux only rescues failed steps"),
         ("planner", "aux only plans; local model executes")]


def _ask(prompt: str, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default and not secret else ""
    try:
        v = (getpass.getpass if secret else input)(f"{prompt}{suffix}: ").strip()
    except EOFError:
        v = ""
    return v or default


def _menu(title: str, options: list[tuple[str, str]], default: int = 1) -> int:
    print(f"\n{title}")
    for i, (a, b) in enumerate(options, 1):
        print(f"  {i}) {a:<20} {b}")
    while True:
        v = _ask("Choose", str(default))
        if v.isdigit() and 1 <= int(v) <= len(options):
            return int(v) - 1


def _test_llm(llm: LLM) -> bool:
    try:
        out = llm.chat([{"role": "user", "content": "Reply with the single word OK."}], max_tokens=16)
        return bool(out.strip())
    except Exception as e:
        say("warn", f"test call failed: {str(e)[:200]}")
        return False


def run(interactive: bool = True) -> dict:
    C.ensure_dirs()
    cfg = C.load_config()
    hw = hardware.detect()
    say("info", "Hardware: " + hardware.summary(hw))
    secrets: dict = {}

    # ---- 1. local model + context window ------------------------------------------------
    if interactive:
        i = _menu("Local model (Ollama, downloaded automatically if missing):", MODELS + [("custom", "enter any ollama tag")])
        name = MODELS[i][0] if i < len(MODELS) else _ask("Ollama model tag", "qwen2.5:3b")
    else:
        name = os.environ.get("NOMAD_MODEL", cfg["model"]["name"])
    cfg["model"]["name"] = name
    rec = hardware.recommend(hw, name)
    for n in rec["notes"]:
        say("warn", n)
    if interactive:
        v = _ask(f"Context window in tokens ('auto' = {rec['num_ctx']} for this machine)", "auto")
        cfg["model"]["num_ctx"] = int(v) if v.isdigit() else None
    else:
        v = os.environ.get("NOMAD_NUM_CTX", "auto")
        cfg["model"]["num_ctx"] = int(v) if v.isdigit() else None
    say("ok", f"model={name}  context={'auto (' + str(rec['num_ctx']) + ')' if not cfg['model']['num_ctx'] else cfg['model']['num_ctx']}")

    # ---- 2. auxiliary model --------------------------------------------------------------
    aux = None
    if interactive:
        j = _menu("Secondary/auxiliary model (optional, stronger cloud model for planning + rescue):",
                  [("none", "local model only"), ("NVIDIA NIM", "needs NVIDIA_API_KEY"), ("Gemini", "needs GEMINI_API_KEY")], 1)
        prov = (None, "nim", "gemini")[j]
    else:
        prov = os.environ.get("NOMAD_AUX_PROVIDER") or (
            "nim" if os.environ.get("NVIDIA_API_KEY") else "gemini" if os.environ.get("GEMINI_API_KEY") else None)
    if prov:
        keyvar = "NVIDIA_API_KEY" if prov == "nim" else "GEMINI_API_KEY"
        key = os.environ.get(keyvar, "")
        if interactive:
            key = _ask(f"{keyvar} (input hidden{'; Enter keeps the one in env' if key else ''})", key, secret=True)
        model = (_ask("Aux model name", DEFAULT_AUX[prov]["name"]) if interactive
                 else os.environ.get("NOMAD_AUX_MODEL", DEFAULT_AUX[prov]["name"]))
        role = ROLES[_menu("Aux role:", ROLES, 1)][0] if interactive else os.environ.get("NOMAD_AUX_ROLE", "planner+fallback")
        if key:
            os.environ[keyvar] = key
            test = LLM(prov, model, DEFAULT_AUX[prov]["num_ctx"], label="aux")
            ok = _test_llm(test)
            say("ok" if ok else "warn", f"aux model {'reachable' if ok else 'NOT reachable'}")
            if ok or not interactive or _ask("Keep it anyway? (y/N)", "n").lower().startswith("y"):
                secrets[keyvar] = key
                aux = {"provider": prov, "name": model, "num_ctx": DEFAULT_AUX[prov]["num_ctx"], "role": role}
        else:
            say("warn", f"no {keyvar}; skipping aux model")
    cfg["aux"] = aux

    # ---- 3. Neon Postgres ----------------------------------------------------------------
    url = os.environ.get("DATABASE_URL", "")
    if interactive:
        url = _ask("Neon DATABASE_URL (postgresql://...; Enter = local SQLite only)", url, secret=True)
    if url:
        st = Store(url)
        if st.error:
            say("warn", "could not connect to that database; using local SQLite for now")
        else:
            say("ok", "Neon connected")
            secrets["DATABASE_URL"] = url
        st.close()
    else:
        say("warn", "no DATABASE_URL: state will NOT survive Colab resets (local SQLite only)")

    # ---- 4. S3 (optional) ----------------------------------------------------------------
    s3 = None
    if interactive:
        want = _ask("\nConfigure S3-compatible blob storage? (y/N)", "n").lower().startswith("y")
        if want:
            s3 = {"endpoint_url": _ask("Endpoint URL (blank for AWS)", ""), "bucket": _ask("Bucket", ""),
                  "region": _ask("Region", "auto"), "prefix": _ask("Key prefix", "nomad/")}
            secrets["S3_ACCESS_KEY"] = _ask("Access key", "", secret=True)
            secrets["S3_SECRET_KEY"] = _ask("Secret key", "", secret=True)
    elif os.environ.get("S3_BUCKET"):
        s3 = {"endpoint_url": os.environ.get("S3_ENDPOINT", ""), "bucket": os.environ["S3_BUCKET"],
              "region": os.environ.get("S3_REGION", "auto"), "prefix": os.environ.get("S3_PREFIX", "nomad/")}
    if s3 and s3.get("bucket"):
        for k in ("S3_ACCESS_KEY", "S3_SECRET_KEY"):
            if k in secrets:
                os.environ[k] = secrets[k]
        try:
            Blob(s3).list("", 1)
            say("ok", "S3 reachable")
        except Exception as e:
            say("warn", f"S3 test failed: {str(e)[:160]}")
    else:
        s3 = None
    cfg["s3"] = s3

    C.save_config(cfg)
    if secrets:
        C.update_env(secrets)
    say("ok", f"saved {C.CONFIG_PATH}" + (f" and secrets to {C.ENV_PATH} (chmod 600)" if secrets else ""))
    return cfg
