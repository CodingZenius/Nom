"""Headless Chromium through Playwright with human-like interaction.

* Runs in its own thread (Playwright's sync API refuses to run inside Jupyter/Colab's asyncio loop).
* Pages are exposed to small models as: compact text + numbered interactive elements.
* Human-like: curved mouse paths with jitter, per-key typing delays, stepped scrolling, random pauses.
* Two screenshot modes:
    headless : clean full-page capture straight from the browser
    human    : the viewport as a person would see it, with the mouse cursor drawn on top
"""
from __future__ import annotations
import os
import queue
import random
import re
import threading
import time

from . import config as C
from .util import slug

SNAP_JS = """() => {
  const sel = 'a[href],button,input,textarea,select,[role=button],[role=link],[onclick]';
  document.querySelectorAll('[data-nomad-id]').forEach(e => e.removeAttribute('data-nomad-id'));
  const out = []; let n = 0;
  for (const el of document.querySelectorAll(sel)) {
    const r = el.getBoundingClientRect(), st = getComputedStyle(el);
    if (r.width < 2 || r.height < 2 || st.visibility === 'hidden' || st.display === 'none') continue;
    if (el.type === 'hidden') continue;
    if (r.bottom < -400 || r.top > innerHeight * 3) continue;
    n++; el.setAttribute('data-nomad-id', n);
    const label = (el.innerText || el.value || el.getAttribute('aria-label') || el.placeholder || el.title || el.alt || '')
      .trim().replace(/\\s+/g, ' ').slice(0, 80);
    out.push({id: n, tag: el.tagName.toLowerCase(), type: el.type || '', text: label, href: (el.href || '').slice(0, 100)});
    if (n >= 60) break;
  }
  return out;
}"""

STEALTH_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
window.chrome = window.chrome || {runtime: {}};
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
"""

CURSOR_JS = """([x, y]) => {
  let e = document.getElementById('__nomad_cursor');
  if (!e) { e = document.createElement('div'); e.id = '__nomad_cursor';
    e.style.cssText = 'position:fixed;z-index:2147483647;pointer-events:none;width:18px;height:18px;' +
      'border:3px solid #e11;border-radius:50%;background:rgba(255,0,0,.25);transform:translate(-50%,-50%)';
    document.documentElement.appendChild(e); }
  e.style.left = x + 'px'; e.style.top = y + 'px';
}"""


class BrowserError(RuntimeError):
    pass


class Browser:
    def __init__(self, exec_path: str | None = None, human: bool = True):
        self.exec_path, self.human = exec_path, human
        self._q: queue.Queue = queue.Queue()
        self._t: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_err: str | None = None
        self._cursor = (240.0, 200.0)
        self.page = None

    # ------------------------------------------------------------ thread plumbing
    def _ensure(self):
        if self._t and self._t.is_alive():
            return
        self._ready.clear()
        self._start_err = None
        self._t = threading.Thread(target=self._loop, daemon=True, name="nomad-browser")
        self._t.start()
        if not self._ready.wait(120):
            raise BrowserError("browser start timed out")
        if self._start_err:
            raise BrowserError(self._start_err)

    def call(self, op: str, **kw):
        timeout = kw.pop("_timeout", 180)
        self._ensure()
        res: queue.Queue = queue.Queue(1)
        self._q.put((op, kw, res))
        try:
            ok, val = res.get(timeout=timeout)
        except queue.Empty:
            raise BrowserError(f"{op} timed out")
        if not ok:
            raise BrowserError(val)
        return val

    def close(self):
        if self._t and self._t.is_alive():
            self._q.put(("__quit__", {}, queue.Queue(1)))
            self._t.join(timeout=10)

    def _loop(self):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(C.BROWSERS_DIR)
        try:
            from playwright.sync_api import sync_playwright
            self._pw = sync_playwright().start()
            kw = dict(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu",
                                           "--disable-blink-features=AutomationControlled"])
            if self.exec_path:
                kw["executable_path"] = self.exec_path
            self._br = self._pw.chromium.launch(**kw)
            plat = "X11; Linux aarch64" if C.ARCH == "arm64" else "X11; Linux x86_64"
            ua = f"Mozilla/5.0 ({plat}) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{self._br.version} Safari/537.36"
            self._ctx = self._br.new_context(viewport={"width": 1366, "height": 768}, user_agent=ua, locale="en-US")
            self._ctx.add_init_script(STEALTH_JS)
            self._ctx.on("page", lambda pg: setattr(self, "page", pg))
            self.page = self._ctx.new_page()
            self.page.set_default_timeout(30000)
        except Exception as e:
            self._start_err = f"{type(e).__name__}: {str(e)[:300]}"
            self._ready.set()
            return
        self._ready.set()
        while True:
            op, kw, res = self._q.get()
            if op == "__quit__":
                break
            try:
                res.put((True, getattr(self, "_op_" + op)(**kw)))
            except Exception as e:
                res.put((False, f"{type(e).__name__}: {str(e)[:300]}"))
        try:
            self._br.close()
            self._pw.stop()
        except Exception:
            pass

    # ------------------------------------------------------------ human behaviour
    def _pause(self, a=0.15, b=0.6):
        if self.human:
            time.sleep(random.uniform(a, b))

    def _move(self, p, x, y):
        sx, sy = self._cursor
        if not self.human:
            p.mouse.move(x, y)
        else:
            cx = (sx + x) / 2 + random.uniform(-80, 80)
            cy = (sy + y) / 2 + random.uniform(-60, 60)
            n = random.randint(14, 28)
            for i in range(1, n + 1):
                t = i / n
                t = t * t * (3 - 2 * t)  # smoothstep easing
                bx = (1 - t) ** 2 * sx + 2 * (1 - t) * t * cx + t * t * x
                by = (1 - t) ** 2 * sy + 2 * (1 - t) * t * cy + t * t * y
                p.mouse.move(bx + random.uniform(-1, 1), by + random.uniform(-1, 1))
                time.sleep(random.uniform(0.004, 0.014))
            p.mouse.move(x, y)
        self._cursor = (x, y)

    def _page(self):
        if self.page is None or self.page.is_closed():
            self.page = self._ctx.new_page()
        return self.page

    def _settle(self, p):
        try:
            p.wait_for_load_state("domcontentloaded", timeout=8000)
            p.wait_for_load_state("networkidle", timeout=4000)
        except Exception:
            pass
        self._pause(0.3, 0.9)

    def _el(self, p, id):
        el = p.query_selector(f'[data-nomad-id="{int(id)}"]')
        if not el:
            raise BrowserError(f"no element with id {id}; call web_read to refresh the element list")
        return el

    def _human_click(self, p, el):
        el.scroll_into_view_if_needed(timeout=5000)
        bb = el.bounding_box()
        if not bb:
            el.click()
            return
        x = bb["x"] + bb["width"] * random.uniform(0.3, 0.7)
        y = bb["y"] + bb["height"] * random.uniform(0.3, 0.7)
        self._move(p, x, y)
        self._pause(0.08, 0.3)
        p.mouse.down()
        time.sleep(random.uniform(0.03, 0.11) if self.human else 0)
        p.mouse.up()

    # ------------------------------------------------------------ view
    def _view(self, offset: int = 0, maxc: int = 1800) -> str:
        p = self._page()
        text = p.evaluate("() => document.body ? document.body.innerText : ''") or ""
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        els = p.evaluate(SNAP_JS)
        chunk = text[offset:offset + maxc]
        lines = [f"URL: {p.url}", f"TITLE: {p.title()}",
                 f"TEXT [{offset}-{offset + len(chunk)} of {len(text)}]:", chunk,
                 "ELEMENTS (use the number with web_click / web_type):"]
        for e in els[:40]:
            kind = e["tag"] + (f"/{e['type']}" if e["type"] else "")
            lines.append(f"[{e['id']}] {kind} {e['text']!r}" + (f" -> {e['href']}" if e["href"] else ""))
        return "\n".join(lines)

    # ------------------------------------------------------------ operations (run in worker thread)
    def _op_goto(self, url: str):
        if not re.match(r"^[a-z]+://", url):
            url = "https://" + url
        p = self._page()
        p.goto(url, wait_until="domcontentloaded", timeout=45000)
        self._settle(p)
        return self._view()

    def _op_read(self, offset: int = 0):
        return self._view(offset=int(offset))

    def _op_click(self, id):
        p = self._page()
        self._human_click(p, self._el(p, id))
        self._settle(p)
        return self._view()

    def _op_type(self, id, text: str, submit: bool = False):
        p = self._page()
        el = self._el(p, id)
        self._human_click(p, el)
        el.fill("")
        if self.human and len(text) <= 160:
            for ch in text:
                p.keyboard.type(ch)
                time.sleep(random.uniform(0.035, 0.15))
                if ch in " .,;" and random.random() < 0.15:
                    time.sleep(random.uniform(0.2, 0.5))
        else:
            el.fill(text)
        if submit:
            self._pause(0.2, 0.6)
            p.keyboard.press("Enter")
            self._settle(p)
        return self._view()

    def _op_scroll(self, direction: str = "down", amount: int = 600):
        p = self._page()
        amt = int(amount) * (1 if direction != "up" else -1)
        steps = random.randint(4, 8) if self.human else 1
        for _ in range(steps):
            p.mouse.wheel(0, amt / steps)
            time.sleep(random.uniform(0.03, 0.12) if self.human else 0)
        self._pause(0.2, 0.5)
        return self._view()

    def _op_key(self, key: str):
        p = self._page()
        p.keyboard.press(key)
        self._settle(p)
        return self._view()

    def _op_back(self):
        p = self._page()
        p.go_back(wait_until="domcontentloaded")
        self._settle(p)
        return self._view()

    def _op_shot(self, mode: str = "headless", full: bool = False, name: str = ""):
        p = self._page()
        mode = "human" if mode == "human" else "headless"
        d = C.SHOTS_DIR / mode
        d.mkdir(parents=True, exist_ok=True)
        fn = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{slug(name or p.title() or 'page')}.png"
        if mode == "human":
            p.evaluate(CURSOR_JS, list(self._cursor))
            p.screenshot(path=str(fn), full_page=False)
            p.evaluate("() => { const e = document.getElementById('__nomad_cursor'); if (e) e.remove(); }")
        else:
            p.screenshot(path=str(fn), full_page=bool(full))
        return str(fn)
