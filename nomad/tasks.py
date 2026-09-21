"""Task manager: turns a goal into small persistent chunks a small model can finish reliably.

Every state change is written to the database, so a killed process (Colab recycle, CPU
migration, crash) resumes exactly where it stopped: `running` tasks are reset to `pending`.
"""
from __future__ import annotations
import uuid
from dataclasses import dataclass, field

MAX_TASKS = 12
MAX_INSTR = 600
MAX_TITLE = 80
MARK = {"done": "✔", "failed": "✘", "running": "▶", "skipped": "⊘", "replaced": "↻", "pending": "·"}


@dataclass
class Task:
    id: str
    goal_id: str
    pos: int
    title: str
    instruction: str
    status: str = "pending"
    result: str = ""
    attempts: int = 0
    deps: list = field(default_factory=list)


class TaskManager:
    def __init__(self, store):
        self.s = store

    # ------------------------------------------------------------ creation
    def new_goal(self, text: str) -> str:
        gid = "g" + uuid.uuid4().hex[:8]
        self.s.goal_add(gid, text)
        return gid

    @staticmethod
    def normalize(obj) -> list[dict]:
        """Validate/clean planner output into [{title, instruction, depends_on:[1-based idx]}]."""
        if isinstance(obj, dict):
            items = obj.get("tasks") or obj.get("steps") or obj.get("plan") or []
        elif isinstance(obj, list):
            items = obj
        else:
            return []
        out = []
        for it in items[:MAX_TASKS]:
            if isinstance(it, str):
                it = {"title": it[:MAX_TITLE], "instruction": it}
            if not isinstance(it, dict):
                continue
            title = str(it.get("title") or it.get("name") or "").strip()
            ins = str(it.get("instruction") or it.get("description") or it.get("task") or title).strip()
            if not (title or ins):
                continue
            raw = it.get("depends_on") or []
            deps = [int(d) for d in raw if str(d).isdigit()] if isinstance(raw, list) else []
            out.append({"title": (title or ins)[:MAX_TITLE], "instruction": ins[:MAX_INSTR], "depends_on": deps})
        return out

    def add_tasks(self, goal_id: str, specs: list[dict]) -> list[str]:
        existing = self.s.tasks_of(goal_id)
        base = max((t["pos"] for t in existing), default=-1) + 1
        ids = [f"t{uuid.uuid4().hex[:8]}" for _ in specs]
        for i, sp in enumerate(specs):
            deps = [ids[d - 1] for d in sp.get("depends_on", []) if 1 <= d <= i]  # earlier tasks only
            self.s.task_add({"id": ids[i], "goal_id": goal_id, "pos": base + i, "title": sp["title"],
                             "instruction": sp["instruction"], "deps": deps})
        return ids

    def replace_pending(self, goal_id: str, specs: list[dict]) -> None:
        self.s.tasks_delete_pending(goal_id)
        self.add_tasks(goal_id, specs)

    # ------------------------------------------------------------ queries
    def tasks(self, goal_id: str) -> list[Task]:
        return [Task(**{k: v for k, v in d.items() if k in Task.__dataclass_fields__})
                for d in self.s.tasks_of(goal_id)]

    def reset_running(self, goal_id: str) -> None:
        for t in self.tasks(goal_id):
            if t.status == "running":
                self.s.task_set(t.id, status="pending")

    def next_ready(self, goal_id: str) -> Task | None:
        ts = self.tasks(goal_id)
        status = {t.id: t.status for t in ts}
        for t in ts:
            if t.status != "pending":
                continue
            ds = [status.get(d, "done") for d in t.deps]
            if any(s in ("failed", "skipped") for s in ds):
                self.s.task_set(t.id, status="skipped", result="dependency failed")
                status[t.id] = "skipped"
                continue
            if all(s in ("done", "replaced") for s in ds):
                return t
        return None

    def set(self, task_id: str, **f) -> None:
        self.s.task_set(task_id, **f)

    def render(self, goal_id: str) -> str:
        ts = self.tasks(goal_id)
        return "\n".join(f"{i + 1}{MARK.get(t.status, '·')} {t.title}" for i, t in enumerate(ts)) or "(no tasks)"

    def progress(self, goal_id: str) -> str:
        ts = self.tasks(goal_id)
        return f"{sum(t.status == 'done' for t in ts)}/{len(ts)} done"
