"""Idempotent environment bootstrap. Every step is "check first, download only if missing".

Order of preference for each component:
  Ollama    : already runnable on PATH  -> per-arch copy in NOMAD_HOME -> download tarball -> install.sh (root)
  Model     : present in `ollama list`  -> `ollama pull`
  Chromium  : Playwright bundle launches -> system chromium/chrome -> `playwright install chromium`
"""
from __future__ import annotations
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from . import config as C
from .ui import say

OLLAMA_BIN = C.OLLAMA_DIR / "bin" / "ollama"


# ---------------------------------------------------------------- python deps
def pip_install(*pkgs: str) -> None:
    say("setup", "pip install " + " ".join(pkgs))
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"pip failed: {r.stderr[-400:]}")


def ensure_py(module: str, pkg: str | None = None) -> None:
    if importlib.util.find_spec(module) is None:
        pip_install(pkg or module)
        importlib.invalidate_caches()


# ---------------------------------------------------------------- ollama
def _is_local() -> bool:
    return urlparse(C.OLLAMA_URL).hostname in ("127.0.0.1", "localhost", "0.0.0.0", "::1")


def ollama_env() -> dict:
    e = os.environ.copy()
    u = urlparse(C.OLLAMA_URL)
    e.update({
        "OLLAMA_MODELS": str(C.MODELS_DIR),
        "OLLAMA_HOST": f"{u.hostname}:{u.port or 11434}",
        "OLLAMA_NUM_PARALLEL": "1",
        "OLLAMA_MAX_LOADED_MODELS": "1",
    })
    return e


def _runs(binpath: Path) -> bool:
    try:
        r = subprocess.run([str(binpath), "--version"], capture_output=True, text=True,
                           timeout=30, env=ollama_env())
        return r.returncode == 0
    except Exception:  # includes "Exec format error" after a CPU-arch change
        return False


def find_ollama() -> Path | None:
    cands = [OLLAMA_BIN]
    w = shutil.which("ollama")
    if w:
        cands.insert(0, Path(w))
    for c in cands:
        if c.exists() and _runs(c):
            return c
    return None


def _download(url: str, dest: Path, label: str) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "nomad-agent"})
    last = 0.0
    with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            got += len(b)
            if time.time() - last > 4:
                last = time.time()
                say("dl", f"{label}: {got / 1e6:.0f}" + (f"/{total / 1e6:.0f}" if total else "") + " MB")


def _extract(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    n = archive.name
    if n.endswith((".tgz", ".tar.gz")):
        with tarfile.open(archive) as t:
            t.extractall(dest)
        return
    if n.endswith(".tar.zst"):
        if shutil.which("zstd") and shutil.which("tar"):
            r = subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(dest)],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return
        ensure_py("zstandard")
        import zstandard
        with open(archive, "rb") as fh, zstandard.ZstdDecompressor().stream_reader(fh) as rd:
            with tarfile.open(fileobj=rd, mode="r|") as t:
                t.extractall(dest)
        return
    raise RuntimeError(f"unknown archive type {n}")


def install_ollama() -> Path:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("automatic Ollama install supports Linux only; install it from ollama.com")
    if C.ARCH not in ("amd64", "arm64"):
        raise RuntimeError(f"unsupported CPU architecture: {C.ARCH}")
    tmp = C.HOME / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    a = C.ARCH
    urls = [
        f"https://github.com/ollama/ollama/releases/latest/download/ollama-linux-{a}.tar.zst",
        f"https://ollama.com/download/ollama-linux-{a}.tar.zst",
        f"https://github.com/ollama/ollama/releases/latest/download/ollama-linux-{a}.tgz",
        f"https://ollama.com/download/ollama-linux-{a}.tgz",
    ]
    for u in urls:
        arc = tmp / u.rsplit("/", 1)[-1]
        try:
            say("setup", f"downloading Ollama ({a}) from {urlparse(u).netloc}")
            _download(u, arc, "ollama")
            shutil.rmtree(C.OLLAMA_DIR, ignore_errors=True)
            _extract(arc, C.OLLAMA_DIR)
            if OLLAMA_BIN.exists():
                OLLAMA_BIN.chmod(0o755)
                arc.unlink(missing_ok=True)
                break
        except Exception as e:
            say("warn", f"ollama download failed ({type(e).__name__}: {str(e)[:120]})")
        arc.unlink(missing_ok=True)
    else:
        if getattr(os, "geteuid", lambda: 1)() == 0 and shutil.which("curl"):
            say("setup", "falling back to official install.sh")
            subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
    b = find_ollama()
    if not b:
        raise RuntimeError("Ollama install failed (check network / disk space)")
    say("ok", f"ollama ready: {b}")
    return b


def ollama_up(timeout: float = 2.0) -> bool:
    import requests
    try:
        return requests.get(f"{C.OLLAMA_URL}/api/tags", timeout=timeout).status_code == 200
    except Exception:
        return False


def start_ollama(binp: Path, wait: int = 90) -> str:
    if ollama_up():
        return "already running"
    if not _is_local():
        raise RuntimeError(f"OLLAMA_URL {C.OLLAMA_URL} is not reachable")
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(C.LOGS_DIR / "ollama.log", "ab")
    subprocess.Popen([str(binp), "serve"], env=ollama_env(), stdout=log, stderr=log,
                     start_new_session=True)
    t0 = time.time()
    while time.time() - t0 < wait:
        if ollama_up():
            say("ok", "ollama server started")
            return "started"
        time.sleep(1)
    raise RuntimeError("ollama server did not come up; see " + str(C.LOGS_DIR / "ollama.log"))


def has_model(name: str) -> bool:
    import requests
    try:
        tags = requests.get(f"{C.OLLAMA_URL}/api/tags", timeout=10).json().get("models", [])
    except Exception:
        return False
    names = {m.get("name") for m in tags} | {m.get("model") for m in tags}
    return name in names or (":" not in name and f"{name}:latest" in names)


def pull_model(name: str, retries: int = 3) -> None:
    import requests
    for attempt in range(1, retries + 1):
        try:
            say("dl", f"pulling model {name} (attempt {attempt}/{retries})")
            last, shown = 0.0, ""
            with requests.post(f"{C.OLLAMA_URL}/api/pull", json={"model": name, "stream": True},
                               stream=True, timeout=(10, 600)) as r:
                for line in r.iter_lines():
                    if not line:
                        continue
                    d = json.loads(line)
                    if d.get("error"):
                        raise RuntimeError(d["error"])
                    st = d.get("status", "")
                    if d.get("total") and d.get("completed"):
                        st += f" {100 * d['completed'] / d['total']:.0f}%"
                    if (st != shown and time.time() - last > 3) or st == "success":
                        last, shown = time.time(), st
                        say("dl", st)
            if has_model(name):
                say("ok", f"model {name} ready")
                return
        except Exception as e:
            say("warn", f"pull failed: {str(e)[:160]}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"could not pull model {name}")


# ---------------------------------------------------------------- chromium / playwright
_LAUNCH_TEST = (
    "from playwright.sync_api import sync_playwright as s\n"
    "p=s().start()\n"
    "b=p.chromium.launch({kw}args=['--no-sandbox','--disable-dev-shm-usage'])\n"
    "b.close();p.stop()\n"
)


def browser_env() -> dict:
    e = os.environ.copy()
    e["PLAYWRIGHT_BROWSERS_PATH"] = str(C.BROWSERS_DIR)
    return e


def _short_err(err: str) -> str:
    """Drop Playwright's box-drawing banner and keep the informative lines."""
    keep = [ln.strip() for ln in (err or "").splitlines()
            if ln.strip() and not any(ch in ln for ch in "╔╗╚╝║═")]
    return " | ".join(keep[:2])[:240] or "unknown error"


def pw_launch_ok(exec_path: str | None = None) -> tuple[bool, str]:
    """Test-launch in a subprocess (also sidesteps Colab's running asyncio loop)."""
    kw = f"executable_path={exec_path!r}, " if exec_path else ""
    try:
        r = subprocess.run([sys.executable, "-c", _LAUNCH_TEST.format(kw=kw)], env=browser_env(),
                           capture_output=True, text=True, timeout=120)
        return r.returncode == 0, _short_err(r.stderr or r.stdout)
    except Exception as e:
        return False, str(e)


def ensure_chromium() -> str | None:
    """Returns None when Playwright's own chromium works, else a system chromium path."""
    ensure_py("playwright")
    ok, _ = pw_launch_ok()
    if ok:
        return None
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        p = shutil.which(name)
        if p and pw_launch_ok(p)[0]:
            say("ok", f"using system browser {p}")
            return p
    say("setup", "downloading Chromium via Playwright")
    r = subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], env=browser_env())
    ok, err = pw_launch_ok()
    if not ok and getattr(os, "geteuid", lambda: 1)() == 0:
        say("setup", "installing system libraries for Chromium")
        subprocess.run([sys.executable, "-m", "playwright", "install-deps", "chromium"], env=browser_env())
        ok, err = pw_launch_ok()
    if not ok:
        raise RuntimeError("chromium unusable: " + err)
    say("ok", "chromium ready")
    return None


# ---------------------------------------------------------------- orchestration
def boot(cfg: dict) -> dict:
    """Make sure everything this config needs exists. Never raises; returns a report."""
    C.ensure_dirs()
    rep: dict = {"errors": [], "ollama": None, "model_ready": False, "chromium": False, "chromium_exec": None}
    m = cfg["model"]
    if m["provider"] == "ollama":
        try:
            ensure_py("requests")
            b = find_ollama() or install_ollama()
            rep["ollama"] = str(b)
            start_ollama(b)
            if not has_model(m["name"]):
                pull_model(m["name"])
            rep["model_ready"] = True
        except Exception as e:
            rep["errors"].append(f"ollama: {e}")
            say("err", f"ollama: {e}")
    try:
        rep["chromium_exec"] = ensure_chromium()
        rep["chromium"] = True
    except Exception as e:
        rep["errors"].append(f"chromium: {e}")
        say("warn", f"browser tools disabled: {e}")
    return rep
