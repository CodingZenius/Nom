"""Tool registry. Signatures are kept short: every token counts for a 3B model."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

NO_BROWSER = "ERROR: browser unavailable (Chromium not installed). Run: python run.py boot"


@dataclass
class Tool:
    name: str
    sig: str
    doc: str
    fn: Callable


class Toolbox:
    def __init__(self, ws, browser=None, memory=None, blob=None):
        self.ws, self.browser, self.memory, self.blob = ws, browser, memory, blob
        self.tools: dict[str, Tool] = {}
        self._register()

    def _add(self, name, sig, doc, fn):
        self.tools[name] = Tool(name, sig, doc, fn)

    def _register(self):
        ws = self.ws
        self._add("shell", "shell(cmd, timeout=120)", "run a bash command in the workspace", ws.run)
        self._add("ls", "ls(path='.', depth=1)", "list files/folders", ws.ls)
        self._add("read_file", "read_file(path, start=1, end=200)", "read numbered lines of a file", ws.read_file)
        self._add("write_file", "write_file(path, content)", "create/overwrite a file with full content", ws.write_file)
        self._add("edit_file", "edit_file(path, old, new)", "replace ONE exact text occurrence in a file", ws.edit_file)
        self._add("revise_file", "revise_file(path, instruction)", "let the model rewrite a small file per instruction", ws.revise_file)
        self._add("search", "search(pattern, path='.')", "regex search inside files", ws.search)
        self._add("web_open", "web_open(url)", "open a page; returns text + numbered elements", self._web("goto"))
        self._add("web_read", "web_read(offset=0)", "read more text of the current page", self._web("read"))
        self._add("web_click", "web_click(id)", "click element number id", self._web("click"))
        self._add("web_type", "web_type(id, text, submit=false)", "type into element id (submit presses Enter)", self._web("type"))
        self._add("web_scroll", "web_scroll(direction='down', amount=600)", "scroll the page", self._web("scroll"))
        self._add("web_key", "web_key(key)", "press a key, e.g. Enter, Escape", self._web("key"))
        self._add("web_back", "web_back()", "go back", self._web("back"))
        self._add("web_shot", "web_shot(mode='headless', full=false, name='')",
                  "screenshot: mode headless (clean page) or human (viewport + cursor)", self._shot)
        if self.memory:
            self._add("remember", "remember(fact, section='Facts')", "save a durable note to Memory.md", self._remember)
        if self.blob and self.blob.enabled:
            self._add("blob_put", "blob_put(path, key='')", "upload a workspace file to S3 storage", self._blob_put)
            self._add("blob_get", "blob_get(key, path='')", "download from S3 storage into the workspace", self._blob_get)
            self._add("blob_list", "blob_list(prefix='')", "list S3 keys", lambda prefix="": "\n".join(self.blob.list(prefix)) or "(empty)")

    # ------------------------------------------------------------ wrappers
    def _web(self, op):
        def fn(**kw):
            if self.browser is None:
                return NO_BROWSER
            return self.browser.call(op, **kw)
        return fn

    def _shot(self, mode="headless", full=False, name=""):
        if self.browser is None:
            return NO_BROWSER
        path = self.browser.call("shot", mode=mode, full=full, name=name)
        extra = ""
        if self.blob and self.blob.enabled:
            try:
                extra = f" (uploaded as {self.blob.put('screenshots/' + Path(path).parent.name + '/' + Path(path).name, path)})"
            except Exception as e:
                extra = f" (S3 upload failed: {e})"
        return f"saved {path}{extra}"

    def _remember(self, fact, section="Facts"):
        return self.memory.remember(fact, section)

    def _blob_put(self, path, key=""):
        p = self.ws.resolve(path)
        return "uploaded as " + self.blob.put(key or f"files/{p.name}", p)

    def _blob_get(self, key, path=""):
        dest = self.ws.resolve(path or Path(key).name)
        return "downloaded to " + self.blob.get(key, dest)

    # ------------------------------------------------------------ interface
    def spec(self) -> str:
        return "\n".join(f"- {t.sig}: {t.doc}" for t in self.tools.values())

    def call(self, name: str, args) -> str:
        t = self.tools.get(name)
        if not t:
            return f"ERROR: unknown tool '{name}'. Valid tools: {', '.join(self.tools)}, done, fail"
        if not isinstance(args, dict):
            return "ERROR: args must be a JSON object"
        try:
            return str(t.fn(**args))
        except TypeError as e:
            return f"ERROR: bad arguments ({e}). Usage: {t.sig}"
        except PermissionError as e:
            return f"ERROR: {e}"
        except Exception as e:  # tool failures are observations, not crashes
            return f"ERROR: {type(e).__name__}: {e}"
