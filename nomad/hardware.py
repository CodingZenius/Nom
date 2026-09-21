"""Detect the current machine (CPU, RAM, cgroup limits, GPU, Colab) and derive settings.

Nothing derived here is persisted as a setting: it is recomputed on every boot, which is
what lets the same codebase move between machines with different CPUs.
"""
from __future__ import annotations
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as C

_FLAGS = {"avx", "avx2", "avx512f", "fma", "f16c", "neon", "asimd", "sve", "amx_tile"}


def _read(p: str) -> str:
    try:
        return Path(p).read_text()
    except Exception:
        return ""


def _cgroup_cpus():
    t = _read("/sys/fs/cgroup/cpu.max").split()
    if len(t) == 2 and t[0] != "max":
        try:
            return int(t[0]) / int(t[1])
        except Exception:
            pass
    q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").strip()
    p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us").strip()
    if q and p and q != "-1":
        try:
            return int(q) / int(p)
        except Exception:
            pass
    return None


def _cgroup_mem():
    pairs = (("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
             ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"))
    for lim, cur in pairs:
        raw = _read(lim).strip()
        if raw and raw != "max":
            try:
                v = int(raw)
                if v < (1 << 60):
                    used = _read(cur).strip()
                    return v, (int(used) if used else None)
            except Exception:
                pass
    return None, None


def detect() -> dict:
    cpuinfo = _read("/proc/cpuinfo")
    model, flags = "", set()
    for line in cpuinfo.splitlines():
        k, _, v = line.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k in ("model name", "hardware", "cpu model") and not model:
            model = v
        if k in ("flags", "features"):
            flags.update(v.split())
    model = model or platform.processor() or platform.machine()
    interesting = sorted(flags & _FLAGS)

    try:
        cores = len(os.sched_getaffinity(0))
    except AttributeError:
        cores = os.cpu_count() or 1
    q = _cgroup_cpus()
    eff = min(cores, q) if q else cores
    eff_cores = max(1, int(eff))

    mi = {}
    for line in _read("/proc/meminfo").splitlines():
        k, _, v = line.partition(":")
        parts = v.split()
        if parts:
            mi[k] = int(parts[0]) * 1024
    total = mi.get("MemTotal", 0)
    avail = mi.get("MemAvailable", mi.get("MemFree", 0))
    lim, used = _cgroup_mem()
    if lim:
        total = min(total, lim) if total else lim
        if used is not None:
            avail = min(avail, max(0, lim - used)) if avail else max(0, lim - used)

    gpu = None
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
            gpu = out[0].strip() if out else None
        except Exception:
            gpu = None

    hw = {
        "arch": C.ARCH,
        "cpu_model": model,
        "cpu_flags": interesting,
        "cores": cores,
        "eff_cores": eff_cores,
        "ram_total_gb": round(total / 2**30, 2),
        "ram_avail_gb": round(avail / 2**30, 2),
        "gpu": gpu,
        "colab": bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU")
                      or "google.colab" in sys.modules),
        "root": hasattr(os, "geteuid") and os.geteuid() == 0,
        "os": platform.platform(),
        "python": platform.python_version(),
    }
    hw["fingerprint"] = fingerprint(hw)
    return hw


def fingerprint(hw: dict) -> str:
    key = [hw["arch"], hw["cpu_model"], hw["cpu_flags"], hw["eff_cores"],
           round(hw["ram_total_gb"]), hw["gpu"]]
    return hashlib.sha1(json.dumps(key).encode()).hexdigest()[:12]


def summary(hw: dict) -> str:
    gpu = f", GPU {hw['gpu']}" if hw["gpu"] else ""
    colab = " [Colab]" if hw["colab"] else ""
    return (f"{hw['arch']} | {hw['cpu_model']} | {hw['eff_cores']} cores | "
            f"{hw['ram_avail_gb']:.1f}/{hw['ram_total_gb']:.1f} GB RAM free{gpu}{colab}")


def _params_b(name: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*b\b", name.lower().replace(":", " "))
    return float(m.group(1)) if m else 3.0


def recommend(hw: dict, model_name: str, model_max_ctx: int | None = None) -> dict:
    """Pick num_thread and a context window that fits this machine's free RAM."""
    p = _params_b(model_name)
    model_gb = p * 0.62 + 0.4                 # ~q4 weights + runtime overhead
    kv_mb_tok = max(0.03, p * 0.02)           # conservative KV-cache + compute-buffer cost per token
    budget_mb = (hw["ram_avail_gb"] - model_gb - 0.8) * 1024
    notes = []
    if hw["ram_avail_gb"] < model_gb + 0.8:
        notes.append(f"RAM tight: ~{model_gb + 0.8:.1f} GB wanted, {hw['ram_avail_gb']:.1f} GB free "
                     "(consider qwen2.5:1.5b)")
    ctx = budget_mb / kv_mb_tok if budget_mb > 0 else 2048
    cap = 32768 if hw["gpu"] else 8192        # CPU prefill gets slow beyond ~8k
    ctx = min(ctx, cap, model_max_ctx or 32768)
    pow2 = 2 ** int(math.log2(max(ctx, 2048)))
    return {"num_thread": hw["eff_cores"], "num_ctx": max(2048, min(pow2, model_max_ctx or 32768)),
            "notes": notes}
