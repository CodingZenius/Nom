"""Persistent store: Neon Postgres (DATABASE_URL) with an automatic local SQLite fallback.

Tables are prefixed `nomad_` so the agent can share a Neon database with other projects.
Connections are lazy and re-established on failure (Neon suspends idle compute).
"""
from __future__ import annotations
import json
import threading

from . import config as C
from .ui import say
from .util import now_iso

GOAL_COLS = ["id", "goal", "status", "summary", "created_at", "updated_at"]
TASK_COLS = ["id", "goal_id", "pos", "title", "instruction", "status", "result",
             "attempts", "deps", "created_at", "updated_at"]
_TASK_SETTABLE = {"status", "result", "attempts", "title", "instruction"}
_GOAL_SETTABLE = {"status", "summary"}


def _ddl(dialect: str) -> list[str]:
    auto = "BIGSERIAL PRIMARY KEY" if dialect == "postgres" else "INTEGER PRIMARY KEY AUTOINCREMENT"
    return [
        "CREATE TABLE IF NOT EXISTS nomad_kv (k TEXT PRIMARY KEY, v TEXT NOT NULL, updated_at TEXT NOT NULL)",
        f"CREATE TABLE IF NOT EXISTS nomad_events (id {auto}, ts TEXT NOT NULL, kind TEXT NOT NULL, data TEXT)",
        f"CREATE TABLE IF NOT EXISTS nomad_messages (id {auto}, ts TEXT NOT NULL, session TEXT, role TEXT, content TEXT)",
        "CREATE TABLE IF NOT EXISTS nomad_goals (id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL, "
        "summary TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS nomad_tasks (id TEXT PRIMARY KEY, goal_id TEXT NOT NULL, pos INTEGER NOT NULL, "
        "title TEXT, instruction TEXT, status TEXT NOT NULL, result TEXT, attempts INTEGER NOT NULL DEFAULT 0, "
        "deps TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS nomad_tasks_goal ON nomad_tasks (goal_id, pos)",
    ]


class Store:
    def __init__(self, url: str | None = None):
        self.url = url
        self.dialect = "postgres" if url else "sqlite"
        self.error: str | None = None
        self._conn = None
        self._lock = threading.RLock()
        self._init()

    # ------------------------------------------------------------ plumbing
    def _connect(self):
        if self.dialect == "postgres":
            try:
                import psycopg
            except ImportError:
                from .bootstrap import ensure_py
                ensure_py("psycopg", "psycopg[binary]>=3.1")
                import psycopg
            # prepare_threshold=None: safe behind Neon's pgbouncer pooler
            return psycopg.connect(self.url, autocommit=True, connect_timeout=15, prepare_threshold=None)
        import sqlite3
        C.HOME.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(C.SQLITE_PATH), check_same_thread=False, isolation_level=None, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")
        return c

    def _init(self):
        try:
            for stmt in _ddl(self.dialect):
                self._run(stmt)
        except Exception as e:
            if self.dialect == "postgres":
                self.error = f"{type(e).__name__}: {str(e)[:200]}"
                say("warn", f"Neon unreachable ({self.error}); falling back to local SQLite (NOT durable on Colab)")
                self.dialect, self._conn = "sqlite", None
                for stmt in _ddl("sqlite"):
                    self._run(stmt)
            else:
                raise

    def _run(self, sql: str, params: tuple = (), fetch: str | None = None):
        if self.dialect == "postgres":
            sql = sql.replace("?", "%s")
        with self._lock:
            for attempt in (1, 2):
                try:
                    if self._conn is None:
                        self._conn = self._connect()
                    cur = self._conn.execute(sql, params)
                    if fetch == "all":
                        return cur.fetchall()
                    if fetch == "one":
                        return cur.fetchone()
                    return None
                except Exception:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                    if attempt == 2:
                        raise

    def close(self):
        try:
            if self._conn is not None:
                self._conn.close()
        except Exception:
            pass
        self._conn = None

    # ------------------------------------------------------------ kv
    def kv_get(self, k: str) -> str | None:
        r = self._run("SELECT v FROM nomad_kv WHERE k=?", (k,), "one")
        return r[0] if r else None

    def kv_updated(self, k: str) -> str | None:
        r = self._run("SELECT updated_at FROM nomad_kv WHERE k=?", (k,), "one")
        return r[0] if r else None

    def kv_set(self, k: str, v: str) -> None:
        self._run("INSERT INTO nomad_kv (k, v, updated_at) VALUES (?,?,?) "
                  "ON CONFLICT (k) DO UPDATE SET v=excluded.v, updated_at=excluded.updated_at",
                  (k, v, now_iso()))

    # ------------------------------------------------------------ logs
    def log(self, kind: str, data=None) -> None:
        try:
            self._run("INSERT INTO nomad_events (ts, kind, data) VALUES (?,?,?)",
                      (now_iso(), kind, json.dumps(data, default=str)))
        except Exception:
            pass

    def add_message(self, session: str, role: str, content: str) -> None:
        try:
            self._run("INSERT INTO nomad_messages (ts, session, role, content) VALUES (?,?,?,?)",
                      (now_iso(), session, role, content))
        except Exception:
            pass

    # ------------------------------------------------------------ goals
    def goal_add(self, gid: str, goal: str) -> None:
        t = now_iso()
        self._run("INSERT INTO nomad_goals (id, goal, status, summary, created_at, updated_at) VALUES (?,?,?,?,?,?)",
                  (gid, goal, "planning", "", t, t))

    def goal_set(self, gid: str, **f) -> None:
        f = {k: v for k, v in f.items() if k in _GOAL_SETTABLE}
        if not f:
            return
        sets = ", ".join(f"{k}=?" for k in f) + ", updated_at=?"
        self._run(f"UPDATE nomad_goals SET {sets} WHERE id=?", (*f.values(), now_iso(), gid))

    def goal_get(self, gid: str) -> dict | None:
        r = self._run(f"SELECT {', '.join(GOAL_COLS)} FROM nomad_goals WHERE id=?", (gid,), "one")
        return dict(zip(GOAL_COLS, r)) if r else None

    def goal_unfinished(self) -> dict | None:
        r = self._run(f"SELECT {', '.join(GOAL_COLS)} FROM nomad_goals WHERE status IN ('planning','running','paused') "
                      "ORDER BY created_at DESC LIMIT 1", (), "one")
        return dict(zip(GOAL_COLS, r)) if r else None

    def goals_recent(self, n: int = 3) -> list[dict]:
        rows = self._run(f"SELECT {', '.join(GOAL_COLS)} FROM nomad_goals ORDER BY created_at DESC LIMIT {int(n)}",
                         (), "all") or []
        return [dict(zip(GOAL_COLS, r)) for r in rows]

    # ------------------------------------------------------------ tasks
    def task_add(self, t: dict) -> None:
        now = now_iso()
        self._run(f"INSERT INTO nomad_tasks ({', '.join(TASK_COLS)}) VALUES ({', '.join('?' * len(TASK_COLS))})",
                  (t["id"], t["goal_id"], t["pos"], t["title"], t["instruction"], t.get("status", "pending"),
                   t.get("result", ""), t.get("attempts", 0), json.dumps(t.get("deps", [])), now, now))

    def task_set(self, tid: str, **f) -> None:
        f = {k: v for k, v in f.items() if k in _TASK_SETTABLE}
        if not f:
            return
        sets = ", ".join(f"{k}=?" for k in f) + ", updated_at=?"
        self._run(f"UPDATE nomad_tasks SET {sets} WHERE id=?", (*f.values(), now_iso(), tid))

    def tasks_of(self, goal_id: str) -> list[dict]:
        rows = self._run(f"SELECT {', '.join(TASK_COLS)} FROM nomad_tasks WHERE goal_id=? ORDER BY pos",
                         (goal_id,), "all") or []
        out = []
        for r in rows:
            d = dict(zip(TASK_COLS, r))
            try:
                d["deps"] = json.loads(d["deps"] or "[]")
            except Exception:
                d["deps"] = []
            out.append(d)
        return out

    def tasks_delete_pending(self, goal_id: str) -> None:
        self._run("DELETE FROM nomad_tasks WHERE goal_id=? AND status='pending'", (goal_id,))
