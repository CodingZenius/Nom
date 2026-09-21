# Nomad Agent

A small-model AI agent that **keeps working when the machine underneath it changes**: different CPU,
different architecture, a recycled Colab VM. The codebase is the constant; everything else is
detected, re-verified and re-tuned on every boot, and all state lives in Neon Postgres.

Default brain: **Ollama + Qwen2.5 3B**, installed automatically if missing.

```
 goal ──► Planner ──► TaskManager (small persistent chunks, in Neon)
              │              │
              │              ▼
              │        Agent (ReAct, JSON actions, context-window aware)
              │              │
              ▼              ▼
   aux model (NIM/Gemini)   Toolbox ─ shell · files · edit/revise · browser · screenshots · memory · S3
   plans + rescues          
   (optional)               LLM: Ollama qwen2.5:3b (local)  ·  Memory.md + Personality.md  ·  Neon
```

## Quick start

```bash
pip install -r requirements.txt
python run.py onboard        # pick model, context window, aux model, Neon URL, S3
python run.py boot           # downloads Ollama, qwen2.5:3b, Chromium if missing
python run.py chat           # or:  python run.py run "build a CLI todo app in python and test it"
```

`python run.py doctor` is a read-only health check. `python run.py resume` continues an interrupted goal.

### Docker
```bash
docker build -t nomad .
docker run -it --rm -v nomad-data:/data -v "$PWD/workspace":/workspace \
  -e DATABASE_URL="postgresql://..." nomad            # first boot downloads Ollama/model/Chromium into /data
```

### Google Colab (no persistence between sessions)
```python
# cell 1
!git clone <your repo or unzip nomad-agent.zip> && cd nomad-agent && pip -q install -r requirements.txt
# cell 2  (use Colab Secrets or paste; DATABASE_URL is the only thing you must supply)
import os; os.environ["DATABASE_URL"] = "postgresql://...neon.tech/neondb?sslmode=require"
os.environ["GEMINI_API_KEY"] = "..."          # optional aux model
# cell 3
!cd nomad-agent && python run.py onboard --yes && python run.py run "your goal here"
```
On a fresh VM the agent restores `config.json`, `Memory.md`, `Personality.md` and any unfinished goal
from Neon, re-downloads what is missing, and carries on. Add `S3_*` variables to keep screenshots and
snapshots in blob storage too. Colab hint: pick a GPU runtime and the context window auto-scales to 32k.

## How your 10 requirements map to code

| # | Requirement | Where |
|---|---|---|
| 0 | Ollama + Qwen2.5 3B, auto-install only if missing | `bootstrap.py` (`find_ollama` → `install_ollama` → `pull_model`) |
| 1 | Headless Chromium, download if missing | `bootstrap.ensure_chromium` (Playwright bundle → system chromium → `playwright install`) |
| 2 | Playwright human-like browsing | `browser.py` (curved mouse paths, typing delays, stepped scroll, pauses) |
| 3 | CLI orchestration: write/edit/revise code, navigate FS | `shell.py`, `tools.py` |
| 4 | Headless + human screenshots | `web_shot(mode=...)` |
| 5 | Neon Postgres persistence | `db.py` (SQLite fallback if unreachable) |
| 6 | Memory.md / Personality.md | `memory.py`, `Personality.md` |
| 7 | Onboarding: models, auto context, NIM/Gemini aux | `onboarding.py`, `hardware.recommend` |
| 8 | Task manager (small chunks) | `tasks.py` |
| 9 | Planner | `planner.py` |
| 10 | Optional S3 blob storage | `storage.py` |

## How it survives a CPU change

* **Nothing hardware-derived is saved as a setting.** Thread count and context window are recomputed each
  boot from the live CPU, cgroup CPU/RAM limits and GPU (`hardware.py`). Pin them with `model.num_ctx` /
  `model.num_thread` in `.nomad/config.json` if you prefer.
* **Binaries are stored per architecture** (`.nomad/ollama/amd64`, `.nomad/browsers/arm64`, ...). Moving
  x86 → arm downloads the right set and leaves the other intact. Model weights are architecture-independent
  and shared.
* **Every boot verifies** that Ollama actually executes (catches "Exec format error") and that Chromium launches.
* **A hardware fingerprint** (arch, CPU model, SIMD flags, cores, RAM, GPU) is stored in Neon; a change is
  logged (`nomad_events`) and reported at startup.
* **State is in the database**, not the disk: goals/tasks resume (`running` → `pending`), Memory.md and
  Personality.md are reconciled (DB copy adopted if the local file is stale), config is restored.

## Small-model design choices

* Actions are plain JSON (`{"thought","tool","args"}`), parsed tolerantly (fences, trailing commas, raw newlines, Python-style dicts). No native tool-calling needed.
* Tasks are capped (≤12 per plan, ≤600-char instructions) and oversized ones are split again.
* Each step sees only: goal, plan checklist, last 3 results, current task. Old observations are trimmed to stay inside the context window.
* Failure ladder: retry → retry with the failure reason → aux model (if configured) → one replan → report honestly.
* Web pages are shown as compact text plus **numbered elements**; the model clicks `id 7` instead of writing selectors.

## Screenshots

* `headless` – clean capture straight from the browser (optionally full page).
* `human` – the viewport as a person sees it, with the mouse cursor drawn on top.

Files land in `.nomad/screenshots/{headless,human}/` (and S3 if configured).

## Configuration

`.nomad/config.json` (non-secret) is written by onboarding. Secrets go to `.nomad/.env` (chmod 600) or real
environment variables; they are **never** written to the database. See `.env.example`.

| Key | Meaning |
|---|---|
| `model.name` | Ollama tag (default `qwen2.5:3b`) |
| `model.num_ctx` / `num_thread` | `null` = auto |
| `aux` | `{provider: nim\|gemini, name, num_ctx, role: planner+fallback\|fallback\|planner}` |
| `agent.max_steps`, `task_retries` | per-task tool-call budget, retries |
| `agent.human_browsing` | disable to browse at full speed |
| `agent.unsafe_paths` | allow file tools outside the workspace |
| `s3` | `{endpoint_url, bucket, region, prefix}`; keys via `S3_ACCESS_KEY` / `S3_SECRET_KEY` |

Tests: `python -m unittest discover -s tests -t .` (uses a scripted fake model; no network needed).

## Security notes

The agent can run shell commands. File tools are confined to the workspace, and `shell` has only a small
denylist, so **run it inside Docker/Colab, not on a machine you care about**. Respect the terms of service and
robots rules of sites you browse.

## Known limits (be aware)

* Automatic downloads need outbound network access to github.com / ollama.com / Playwright's CDN. The Ollama
  archive for amd64 is large (bundles CUDA libs).
* A 3B model is capable at small, well-specified steps and will still make mistakes on open-ended work; that is
  why the planner, retries and the optional aux model exist.
* Linux only for auto-install (macOS/Windows: install Ollama yourself, the rest works).
* The Playwright browser runs headless; the "human" screenshot is a viewport capture with a cursor overlay, not a
  desktop capture.

## Ideas for next steps
Vision-based page understanding via the aux model, a FastAPI/HF-Space wrapper, Telegram gateway, encrypted secret sync.
