"""Planner: goal -> small tasks -> execution with retry, aux-model escalation and one replan.

Model routing (aux = optional NVIDIA NIM / Gemini model set during onboarding):
  planning / replanning : aux if role includes "planner", else the local model
  task attempt 1-2      : local model
  task attempt 3        : aux if role includes "fallback" (also used immediately if local backend is down)
"""
from __future__ import annotations
import platform

from . import config as C
from .llm import LLMError
from .tasks import MAX_INSTR, TaskManager
from .ui import say
from .util import extract_json, trunc

PLAN_SYS = """You are the planner for a small AI agent that has these abilities: shell commands, reading/writing/editing files, and browsing the web (open pages, click, type, screenshot).
Break the GOAL into small, concrete tasks. Rules:
- 1 to 8 tasks. Simple goal = 1-2 tasks.
- Each task does ONE thing and needs at most ~6 tool calls. Name exact files/commands/URLs.
- Tasks run in order. Later tasks may use earlier results.
- Last task should verify the outcome (run the code, check the file, look at the page).
Return ONLY JSON: {"tasks":[{"title":"short title","instruction":"precise instruction","depends_on":[]}]}
depends_on holds 1-based numbers of earlier tasks that must succeed first (usually empty)."""

REPLAN_SYS = """You are the planner for a small AI agent. A task failed. Given the goal, what is already done, and the failure, write the REMAINING tasks (a different approach if needed). Same rules and JSON format as before:
{"tasks":[{"title":"...","instruction":"...","depends_on":[]}]}
Return only JSON. If the goal cannot be reached, return {"tasks":[]}."""

MAX_REPLANS = 1


class Planner:
    def __init__(self, llm, aux, tm, agent, store, memory, cfg):
        self.llm, self.aux, self.tm, self.agent, self.s, self.memory, self.cfg = llm, aux, tm, agent, store, memory, cfg

    # ------------------------------------------------------------ routing
    @property
    def _role(self) -> str:
        return ((self.cfg.get("aux") or {}).get("role") or "planner+fallback")

    def _planner_llm(self):
        return self.aux if (self.aux and "planner" in self._role) else self.llm

    def _exec_llm(self, attempt: int, main_down: bool):
        if self.aux and "fallback" in self._role and (attempt >= 2 or main_down):
            return self.aux
        return self.llm

    # ------------------------------------------------------------ planning
    def _ask_plan(self, system: str, user: str) -> list[dict]:
        llm = self._planner_llm()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for _ in range(2):
            try:
                raw = llm.chat(msgs, json_mode=True, max_tokens=1500)
            except LLMError as e:
                if llm is not self.llm:  # aux failed -> try local once
                    llm = self.llm
                    continue
                say("warn", f"planner model error: {e}")
                return []
            specs = TaskManager.normalize(extract_json(raw))
            if specs:
                return specs
            msgs += [{"role": "assistant", "content": trunc(raw, 400)},
                     {"role": "user", "content": 'Invalid. Return ONLY JSON like {"tasks":[{"title":"..","instruction":"..","depends_on":[]}]}'}]
        return []

    def _split_big(self, specs: list[dict]) -> list[dict]:
        out = []
        for sp in specs:
            if len(sp["instruction"]) < MAX_INSTR * 0.9:
                out.append(sp)
                continue
            sub = self._ask_plan(PLAN_SYS, f"GOAL (split into at most 3 smaller tasks):\n{sp['instruction']}")
            out.extend(sub[:3] if sub else [sp])
        return out

    def plan(self, goal_id: str, goal: str) -> list[dict]:
        recent = [g for g in self.s.goals_recent(4) if g["id"] != goal_id][:3]
        hist = "\n".join(f"- {trunc(g['goal'], 100)} [{g['status']}]" for g in recent) or "(none)"
        env = f"{platform.system()} {C.ARCH}, workspace {C.WORKSPACE}"
        specs = self._ask_plan(PLAN_SYS, f"GOAL: {goal}\n\nENVIRONMENT: {env}\nRECENT GOALS:\n{hist}\n\nReturn the JSON plan.")
        if not specs:
            say("warn", "planner produced nothing valid; running the goal as a single task")
            specs = [{"title": trunc(goal, 60), "instruction": goal[:MAX_INSTR], "depends_on": []}]
        specs = self._split_big(specs)
        self.tm.add_tasks(goal_id, specs)
        self.s.goal_set(goal_id, status="running")
        say("plan", "\n" + self.tm.render(goal_id))
        return specs

    def replan(self, goal_id: str, goal: str, failed, reason: str) -> bool:
        ts = self.tm.tasks(goal_id)
        done = "\n".join(f"- {t.title}: {trunc(t.result, 200)}" for t in ts if t.status == "done") or "(nothing)"
        user = (f"GOAL: {goal}\n\nDONE SO FAR:\n{done}\n\nFAILED TASK: {failed.title}\n{failed.instruction}\n"
                f"FAILURE: {trunc(reason, 300)}\n\nWrite the remaining tasks.")
        specs = self._ask_plan(REPLAN_SYS, user)
        if not specs:
            return False
        self.tm.set(failed.id, status="replaced")
        self.tm.replace_pending(goal_id, self._split_big(specs))
        say("plan", "replanned:\n" + self.tm.render(goal_id))
        return True

    # ------------------------------------------------------------ execution
    def _task_prompt(self, goal_id: str, goal: str, t, reason: str) -> str:
        done = [x for x in self.tm.tasks(goal_id) if x.status == "done"][-3:]
        prev = "\n".join(f"- {d.title}: {trunc(d.result, 300)}" for d in done) or "(none yet)"
        p = (f"GOAL: {goal}\nPLAN (✔ done, ▶ current, · pending):\n{self.tm.render(goal_id)}\n"
             f"PREVIOUS RESULTS:\n{prev}\n\nCURRENT TASK: {t.title}\n{t.instruction}\n")
        if reason:
            p += f"\nYOUR PREVIOUS ATTEMPT FAILED: {trunc(reason, 300)}\nUse a different approach.\n"
        return p + "\nWork ONLY on the current task, then call done."

    def _exec_task(self, goal_id: str, goal: str, t) -> tuple[bool, str]:
        tries = 1 + int(self.cfg["agent"].get("task_retries", 2))
        reason, main_down = "", False
        for attempt in range(tries):
            llm = self._exec_llm(attempt, main_down)
            self.tm.set(t.id, attempts=t.attempts + attempt + 1)
            say("task", f"[{self.tm.progress(goal_id)}] {t.title}  (attempt {attempt + 1}/{tries}, {llm.label})")
            res = self.agent.run_step(llm, self._task_prompt(goal_id, goal, t, reason))
            if res.ok:
                return True, res.summary
            reason = res.summary
            if reason.startswith("LLMError") and llm is self.llm:
                main_down = True
            say("warn", f"attempt failed: {trunc(reason, 160)}")
        return False, reason

    def run(self, goal: str | None = None, goal_id: str | None = None) -> dict:
        if goal_id is None:
            goal_id = self.tm.new_goal(goal)
            self.s.add_message("cli", "user", goal)
            self.plan(goal_id, goal)
        else:
            goal = self.s.goal_get(goal_id)["goal"]
            self.tm.reset_running(goal_id)
            self.s.goal_set(goal_id, status="running")
            say("plan", f"resuming: {goal}\n" + self.tm.render(goal_id))
        replans = 0
        try:
            while True:
                t = self.tm.next_ready(goal_id)
                if t is None:
                    break
                self.tm.set(t.id, status="running")
                ok, res = self._exec_task(goal_id, goal, t)
                self.tm.set(t.id, status="done" if ok else "failed", result=res)
                if not ok and replans < MAX_REPLANS:
                    replans += 1
                    self.replan(goal_id, goal, t, res)
        except KeyboardInterrupt:
            self.s.goal_set(goal_id, status="paused")
            say("warn", "paused. Resume with: /resume  (or python run.py resume)")
            return {"goal_id": goal_id, "status": "paused", "summary": ""}
        ts = self.tm.tasks(goal_id)
        failed = [t for t in ts if t.status == "failed"]
        status = "done" if not failed else ("partial" if any(t.status == "done" for t in ts) else "failed")
        summary = self._summarize(goal, ts, status)
        self.s.goal_set(goal_id, status=status, summary=summary)
        self.s.add_message("cli", "assistant", summary)
        say("ok" if status == "done" else "warn", f"{status.upper()}: {summary}")
        return {"goal_id": goal_id, "status": status, "summary": summary}

    def _summarize(self, goal: str, ts, status: str) -> str:
        lines = "\n".join(f"- [{t.status}] {t.title}: {trunc(t.result, 200)}" for t in ts if t.status != "replaced")
        try:
            return self.llm.chat([
                {"role": "system", "content": "Summarize the outcome for the user in 1-3 plain sentences. Mention any failure honestly."},
                {"role": "user", "content": f"GOAL: {goal}\nSTATUS: {status}\nTASKS:\n{lines}"}],
                max_tokens=200, temperature=0.2).strip()
        except Exception:
            return trunc(lines, 600)
