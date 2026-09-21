import json
import unittest
from pathlib import Path

from nomad import config as C
from nomad.agent import Agent
from nomad.db import Store
from nomad.memory import Memory
from nomad.planner import Planner
from nomad.shell import Workspace
from nomad.tasks import TaskManager
from nomad.tools import Toolbox


class FakeLLM:
    """Scripted model: pops one reply per chat() call."""
    provider, name, num_ctx, num_thread, label = "fake", "fake", 8192, None, "main"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append(messages)
        if not self.replies:
            return "{}"
        r = self.replies.pop(0)
        return r if isinstance(r, str) else json.dumps(r)


def act(tool, **args):
    return {"thought": "t", "tool": tool, "args": args}


def build(llm, aux=None, cfg_extra=None):
    C.ensure_dirs()
    store = Store(None)
    mem = Memory(store, llm_getter=lambda: llm)
    mem.restore()
    ws = Workspace(C.WORKSPACE, llm_getter=lambda: llm)
    tools = Toolbox(ws, None, mem, None)
    cfg = C.load_config()
    cfg["agent"]["task_retries"] = 1
    if cfg_extra:
        cfg = C.deep_merge(cfg, cfg_extra)
    tm = TaskManager(store)
    agent = Agent(tools, mem, cfg, store)
    return Planner(llm, aux, tm, agent, store, mem, cfg), store, tm, ws, mem


class TestE2E(unittest.TestCase):
    def test_full_goal(self):
        llm = FakeLLM([
            {"tasks": [{"title": "write script", "instruction": "create hello.py printing hi"},
                       {"title": "run it", "instruction": "run hello.py", "depends_on": [1]}]},
            act("write_file", path="hello.py", content="print('hi')\n"),
            act("done", summary="wrote hello.py"),
            act("shell", cmd="python3 hello.py"),
            act("remember", fact="hello.py prints hi", section="Facts"),
            act("done", summary="ran fine, prints hi"),
            "Created and ran hello.py.",
        ])
        pl, store, tm, ws, mem = build(llm)
        r = pl.run("make hello world")
        self.assertEqual(r["status"], "done", r)
        self.assertTrue((C.WORKSPACE / "hello.py").exists())
        self.assertEqual([t.status for t in tm.tasks(r["goal_id"])], ["done", "done"])
        self.assertIn("hello.py prints hi", mem.memory_text())
        self.assertIn("hello.py prints hi", store.kv_get("file:Memory.md"))   # synced to DB
        self.assertTrue(tm.tasks(r["goal_id"])[1].deps)                       # dependency stored

    def test_resume_after_crash(self):
        llm = FakeLLM([act("done", summary="finished second")])
        pl, store, tm, ws, mem = build(llm)
        gid = tm.new_goal("resume me")
        tm.add_tasks(gid, [{"title": "a", "instruction": "a", "depends_on": []},
                           {"title": "b", "instruction": "b", "depends_on": []}])
        ts = tm.tasks(gid)
        tm.set(ts[0].id, status="done", result="ok")
        tm.set(ts[1].id, status="running")          # process died mid-task
        llm.replies.append("summary")
        r = pl.run(goal_id=gid)
        self.assertEqual(r["status"], "done")
        self.assertEqual(tm.tasks(gid)[1].result, "finished second")
        # and a NEW process would see the same state:
        self.assertEqual(Store(None).goal_get(gid)["status"], "done")

    def test_retry_then_replan(self):
        bad = "not json at all"
        llm = FakeLLM([
            {"tasks": [{"title": "hard", "instruction": "do hard thing"}]},
            bad, bad, bad,                     # attempt 1: 3 invalid replies -> fail
            bad, bad, bad,                     # attempt 2 -> fail
            {"tasks": [{"title": "easy", "instruction": "do easy thing"}]},   # replan
            act("done", summary="easy done"),
            "Did the easy version.",
        ])
        pl, store, tm, ws, mem = build(llm)
        r = pl.run("something hard")
        st = [t.status for t in tm.tasks(r["goal_id"])]
        self.assertEqual(st, ["replaced", "done"])
        self.assertEqual(r["status"], "done")

    def test_aux_escalation_on_third_attempt(self):
        main = FakeLLM([{"tasks": [{"title": "x", "instruction": "x"}]}] + ["bad"] * 6 + ["sum"])
        aux = FakeLLM([act("done", summary="aux rescued")])
        aux.label = "aux"
        # planner role = fallback only, so main plans; task_retries=2 -> 3rd attempt uses aux
        pl, store, tm, ws, mem = build(main, aux, {"aux": {"provider": "nim", "name": "n", "role": "fallback"},
                                                   "agent": {"task_retries": 2}})
        r = pl.run("needs rescue")
        self.assertEqual(tm.tasks(r["goal_id"])[0].result, "aux rescued")

    def test_tools_safety_and_edit(self):
        llm = FakeLLM([])
        pl, store, tm, ws, mem = build(llm)
        tb = pl.agent.tools
        self.assertIn("blocked", tb.call("shell", {"cmd": "rm -rf /"}))
        self.assertIn("outside workspace", tb.call("read_file", {"path": "/etc/passwd"}))
        tb.call("write_file", {"path": "d/x.py", "content": "a = 1\nb = 2\n"})
        self.assertIn("edited", tb.call("edit_file", {"path": "d/x.py", "old": "b = 2", "new": "b = 3"}))
        self.assertIn("not found", tb.call("edit_file", {"path": "d/x.py", "old": "zzz", "new": "y"}))
        self.assertIn("syntax error", tb.call("write_file", {"path": "bad.py", "content": "def (:\n"}))
        self.assertIn("x.py:2", tb.call("search", {"pattern": "b = 3"}))
        self.assertIn("timed out", tb.call("shell", {"cmd": "sleep 5", "timeout": 1}))
        self.assertIn("unknown tool", tb.call("nope", {}))
        self.assertIn("bad arguments", tb.call("ls", {"bogus": 1}))
        self.assertIn("browser unavailable", tb.call("web_open", {"url": "x"}))
        self.assertIn("refused", mem.remember("my api_key = sk-123"))

    def test_revise_file(self):
        llm = FakeLLM(["```python\ndef add(a, b):\n    return a + b\n```"])
        pl, store, tm, ws, mem = build(llm)
        ws.write_file("m.py", "def add(a,b): return a-b\n")
        out = ws.revise_file("m.py", "fix add")
        self.assertIn("revised", out)
        self.assertIn("a + b", (C.WORKSPACE / "m.py").read_text())

    def test_memory_survives_machine_change(self):
        """Simulate a fresh Colab VM: local Memory.md gone, DB still has it."""
        llm = FakeLLM([])
        pl, store, tm, ws, mem = build(llm)
        mem.remember("user prefers tabs", "User")
        C.MEMORY_MD.unlink()
        C.STATE_PATH.unlink(missing_ok=True)
        Memory(Store(None)).restore()
        self.assertIn("user prefers tabs", C.MEMORY_MD.read_text())


if __name__ == "__main__":
    unittest.main()
