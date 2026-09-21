"""One `chat()` interface over Ollama (local default), NVIDIA NIM and Gemini."""
from __future__ import annotations
import json
import os
import time

import requests

from . import config as C

DEFAULT_AUX = {
    "nim": {"name": "meta/llama-3.3-70b-instruct", "num_ctx": 32768},
    "gemini": {"name": "gemini-2.5-flash", "num_ctx": 32768},
}


class LLMError(RuntimeError):
    def __init__(self, msg: str, retryable: bool = False):
        super().__init__(msg)
        self.retryable = retryable


class LLM:
    def __init__(self, provider: str, name: str, num_ctx: int = 8192, num_thread: int | None = None,
                 temperature: float = 0.2, keep_alive: str = "30m", label: str = "main"):
        self.provider, self.name, self.num_ctx = provider, name, int(num_ctx)
        self.num_thread, self.temperature, self.keep_alive, self.label = num_thread, temperature, keep_alive, label
        self.recover = None  # optional callable: try to revive a dead backend (e.g. restart ollama)

    def __repr__(self):
        return f"<LLM {self.label} {self.provider}:{self.name} ctx={self.num_ctx}>"

    # ---------------------------------------------------------------- public
    def chat(self, messages: list[dict], *, json_mode: bool = False, max_tokens: int = 1024,
             temperature: float | None = None, on_token=None) -> str:
        temp = self.temperature if temperature is None else temperature
        recovered = False
        for attempt in range(3):
            try:
                if self.provider == "ollama":
                    return self._ollama(messages, json_mode, max_tokens, temp, on_token)
                if self.provider == "nim":
                    return self._nim(messages, max_tokens, temp)
                if self.provider == "gemini":
                    return self._gemini(messages, json_mode, max_tokens, temp)
                raise LLMError(f"unknown provider {self.provider}")
            except LLMError as e:
                if not e.retryable or attempt == 2:
                    raise
                if self.recover and not recovered:
                    recovered = True
                    try:
                        self.recover()
                    except Exception:
                        pass
                time.sleep(2 * (attempt + 1))
        raise LLMError("unreachable")

    # ---------------------------------------------------------------- providers
    def _ollama(self, messages, json_mode, max_tokens, temp, on_token):
        opts = {"num_ctx": self.num_ctx, "temperature": temp, "num_predict": max_tokens}
        if self.num_thread:
            opts["num_thread"] = self.num_thread
        body = {"model": self.name, "messages": messages, "stream": True,
                "keep_alive": self.keep_alive, "options": opts}
        if json_mode:
            body["format"] = "json"
        try:
            r = requests.post(f"{C.OLLAMA_URL}/api/chat", json=body, stream=True, timeout=(10, 300))
        except requests.RequestException as e:
            raise LLMError(f"ollama unreachable: {e}", retryable=True)
        if r.status_code != 200:
            raise LLMError(f"ollama HTTP {r.status_code}: {r.text[:300]}", retryable=r.status_code >= 500)
        out = []
        try:
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if "error" in d:
                    raise LLMError(f"ollama: {d['error']}")
                piece = d.get("message", {}).get("content", "")
                if piece:
                    out.append(piece)
                    if on_token:
                        on_token(piece)
                if d.get("done"):
                    break
        except requests.RequestException as e:
            raise LLMError(f"ollama stream broke: {e}", retryable=True)
        return "".join(out)

    def _nim(self, messages, max_tokens, temp):
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            raise LLMError("NVIDIA_API_KEY not set")
        base = os.environ.get("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        body = {"model": self.name, "messages": messages, "max_tokens": max_tokens,
                "temperature": temp, "stream": False}
        try:
            r = requests.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"},
                              json=body, timeout=180)
        except requests.RequestException as e:
            raise LLMError(f"nim unreachable: {e}", retryable=True)
        if r.status_code != 200:
            raise LLMError(f"nim HTTP {r.status_code}: {r.text[:300]}", retryable=r.status_code in (429, 500, 502, 503, 504))
        return r.json()["choices"][0]["message"].get("content") or ""

    def _gemini(self, messages, json_mode, max_tokens, temp):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise LLMError("GEMINI_API_KEY not set")
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "user" if m["role"] == "user" else "model"
            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"][0]["text"] += "\n\n" + m["content"]
            else:
                contents.append({"role": role, "parts": [{"text": m["content"]}]})
        gen = {"temperature": temp, "maxOutputTokens": max(max_tokens, 2048)}
        if json_mode:
            gen["responseMimeType"] = "application/json"
        body = {"contents": contents, "generationConfig": gen}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.name}:generateContent"
        try:
            r = requests.post(url, headers={"x-goog-api-key": key}, json=body, timeout=180)
        except requests.RequestException as e:
            raise LLMError(f"gemini unreachable: {e}", retryable=True)
        if r.status_code != 200:
            raise LLMError(f"gemini HTTP {r.status_code}: {r.text[:300]}", retryable=r.status_code in (429, 500, 502, 503, 504))
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError):
            raise LLMError(f"gemini returned no content: {r.text[:200]}")


# -------------------------------------------------------------------- helpers
def ollama_model_ctx(name: str) -> int | None:
    """Ask the running Ollama for the model's trained max context length."""
    try:
        r = requests.post(f"{C.OLLAMA_URL}/api/show", json={"model": name}, timeout=15)
        for k, v in (r.json().get("model_info") or {}).items():
            if k.endswith(".context_length"):
                return int(v)
    except Exception:
        pass
    return None


def build_llms(cfg: dict, rec: dict) -> tuple[LLM, LLM | None]:
    m = cfg["model"]
    main = LLM(m["provider"], m["name"], num_ctx=m.get("num_ctx") or rec["num_ctx"],
               num_thread=m.get("num_thread") or rec["num_thread"],
               temperature=m.get("temperature", 0.2), keep_alive=m.get("keep_alive", "30m"), label="main")
    aux = None
    a = cfg.get("aux")
    if a and a.get("provider") in DEFAULT_AUX:
        d = DEFAULT_AUX[a["provider"]]
        aux = LLM(a["provider"], a.get("name") or d["name"], num_ctx=a.get("num_ctx") or d["num_ctx"],
                  temperature=0.2, label="aux")
    return main, aux
