"""ReAct-style step runner built for small models.

Protocol: the model answers each turn with ONE JSON object
    {"thought": "...", "tool": "<name>", "args": {...}}
(no native tool-calling needed, so any 1-3B model works). `done` / `fail` end the step.
The message list is kept inside the model's context window by trimming old observations.
"""
from __future__ import annotations
import json
import platform
from collections import Counter
from dataclasses import dataclass

from . import config as C
from .llm import LLMError
from .ui import say
from .util import est_tokens, extract_json, trunc

PROTOCOL = """## How to act
Reply with exactly ONE JSON object per turn and nothing else:
{"thought": "<one short sentence>", "tool": "<tool name>", "args": {<arguments>}}
When the CURRENT TASK is complete: {"thought": "...", "tool": "done", "args": {"summary": "<what you did / the result>"}}
If it cannot be done: {"thought": "...", "tool": "fail", "args": {"reason": "<why>"}}
Example: {"thought": "see what files exist", "tool": "ls", "args": {"path": "."}}
Tool output arrives as OBSERVATION. Use one tool per turn. Never guess results; look first."""


@dataclass
class StepResult:
    ok: bool
    summary: str
    steps: int = 0


def parse_action(raw: str):
    obj = extract_json(raw)
    if isinstance(obj, list):
        obj = next((x for x in obj if isinstance(x, dict)), None)
    if not isinstance(obj, dict):
        return None
    tool = obj.get("tool") or obj.get("action") or obj.get("name")
    if not isinstance(tool, str) or not tool:
        return None
    args = obj.get("args") or obj.get("arguments") or obj.get("input") or obj.get("parameters")
    if not isinstance(args, dict):
        # small models often flatten: {"tool":"done","summary":"..."}
        args = {k: v for k, v in obj.items() if k not in ("thought", "tool", "action", "name", "args")}
    return {"thought": str(obj.get("thought", ""))[:300], "tool": tool.strip(), "args": args}


class Agent:
    def __init__(self, toolbox, memory, cfg: dict, store=None):
        self.tools, self.memory, self.cfg, self.store = toolbox, memory, cfg, store

    def _system(self, llm) -> str:
        mem_chars = min(3500, int(llm.num_ctx * 0.25 * 3))
        env = f"Workspace: {C.WORKSPACE} | OS: {platform.system()} {C.ARCH} | Python {platform.python_version()}"
        return (f"{self.memory.context_block(mem_chars)}\n\n## Environment\n{env}\n\n"
                f"## Tools\n{self.tools.spec()}\n\n{PROTOCOL}")

    @staticmethod
    def _fit(msgs: list[dict], budget: int) -> list[dict]:
        def total():
            return sum(est_tokens(m["content"]) for m in msgs)
        i = 2
        while total() > budget and i < len(msgs):          # shrink old observations first
            c = msgs[i]["content"]
            if len(c) > 240:
                msgs[i] = {"role": msgs[i]["role"], "content": c[:200] + " ...[trimmed]"}
            i += 1
        while total() > budget and len(msgs) > 4:          # then drop oldest turn pairs
            del msgs[2:4]
        if total() > budget and len(msgs[1]["content"]) > 1500:
            msgs[1] = {"role": "user", "content": trunc(msgs[1]["content"], 1500)}
        return msgs

    def run_step(self, llm, task_prompt: str, max_steps: int | None = None) -> StepResult:
        max_steps = max_steps or self.cfg["agent"]["max_steps"]
        max_tokens = min(1200, max(256, llm.num_ctx // 4))
        budget = llm.num_ctx - max_tokens - 100
        msgs = [{"role": "system", "content": self._system(llm)}, {"role": "user", "content": task_prompt}]
        bad, seen = 0, Counter()
        for step in range(1, max_steps + 1):
            msgs = self._fit(msgs, budget)
            try:
                raw = llm.chat(msgs, json_mode=True, max_tokens=max_tokens)
            except LLMError as e:
                return StepResult(False, f"LLMError: {e}", step)
            act = parse_action(raw)
            if not act:
                bad += 1
                say("warn", f"step {step}: unparseable reply ({bad}/3)")
                if bad >= 3:
                    return StepResult(False, "model kept returning invalid JSON", step)
                msgs += [{"role": "assistant", "content": trunc(raw, 300)},
                         {"role": "user", "content": 'ERROR: reply with ONE JSON object: {"thought":"..","tool":"..","args":{..}}'}]
                continue
            bad = 0
            tool, args = act["tool"], act["args"]
            say("tool", f"{step}. {tool} {trunc(json.dumps(args, ensure_ascii=False), 140)}")
            if act["thought"]:
                say("think", act["thought"])
            if tool == "done":
                return StepResult(True, str(args.get("summary") or args.get("result") or act["thought"] or "done"), step)
            if tool == "fail":
                return StepResult(False, str(args.get("reason") or act["thought"] or "failed"), step)
            key = (tool, json.dumps(args, sort_keys=True, default=str))
            seen[key] += 1
            if seen[key] >= 4:
                return StepResult(False, f"stuck repeating {tool}", step)
            obs = self.tools.call(tool, args)
            if seen[key] == 2:
                obs += "\nNOTE: you already did exactly this. Do something different or finish."
            if self.store:
                self.store.log("tool", {"tool": tool, "args": args, "obs": trunc(obs, 300)})
            msgs += [{"role": "assistant", "content": json.dumps({"tool": tool, "args": args}, ensure_ascii=False)[:1200]},
                     {"role": "user", "content": "OBSERVATION:\n" + trunc(obs, 1800, tail=400)}]
        return StepResult(False, "step budget exhausted without calling done", max_steps)
