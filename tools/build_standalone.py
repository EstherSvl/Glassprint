"""Build the single-file browser version.

The desktop tool is a Python server with a web page in front of it. On an iPad
there is no server and no terminal, so this script folds the page, its styles,
its scripts and the whole glassprint package into one HTML file that runs
Python in the browser tab instead.

    python tools/build_standalone.py

Writes ``docs/index.html``. That path is not an accident: GitHub Pages offers
to publish a folder called ``docs``, so switching Pages on in the repository
settings puts this page straight onto the web with nothing else to configure.
Any other https host works just as well.

The output is committed to the repository, and ``tests/test_standalone.py``
rebuilds it to check it has not drifted from the sources.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "glassprint" / "web"
PACKAGE = ROOT / "glassprint"
OUTPUT = ROOT / "docs" / "index.html"

# The JavaScript below is a raw string on purpose. Python would otherwise read
# the escapes meant for the browser — turning \n into a real newline and \b
# into a backspace, which silently emits a page that cannot parse.

# cli.py wants typer and server.py wants fastapi; neither exists in the browser
# and nothing the page calls imports them.
SKIP = {"cli.py", "server.py"}

BOOT = r"""
/* Start Python, then hand the page over to app.js.
 *
 * Every step is timed and left on screen. On a tablet there is no console to
 * open when something goes wrong, so the page has to be able to say for itself
 * what finished, how long each part took, and exactly where it stopped — a
 * single glance has to be a usable bug report. */
(function () {
  const backend = window.GlassprintBackends.PyodideBackend;
  window.GlassprintBackend = backend;

  const splash = document.getElementById("boot");
  const note = document.getElementById("boot-note");
  const list = document.getElementById("boot-steps");
  const stuck = document.getElementById("boot-stuck");

  const steps = [];
  const SLOW = 45000;

  function paint() {
    const now = Date.now();
    list.innerHTML = steps
      .map((step) => {
        const seconds = ((step.until || now) - step.at) / 1000;
        const size = step.bytes ? " · " + (step.bytes / 1048576).toFixed(1) + " MB" : "";
        const clock = seconds < 10 ? seconds.toFixed(1) : Math.round(seconds);
        return (
          "<li class='" + (step.until ? "done" : "live") + "'>" +
          step.text + size + " · " + clock + "s</li>"
        );
      })
      .join("");

    const active = steps[steps.length - 1];
    if (active && !active.until && now - active.at > SLOW) {
      stuck.hidden = false;
      stuck.innerHTML =
        "<strong>This step is taking far longer than it should.</strong><br />" +
        "Reloading is safe and quick — whatever has downloaded already is kept.";
    }
  }

  backend.onProgress = (text) => {
    const now = Date.now();
    const previous = steps[steps.length - 1];
    if (previous) previous.until = now;
    steps.push({ text: text, at: now });
    paint();
  };
  // Bytes land in bursts; the repaint on the interval is what shows them.
  backend.onBytes = (count) => {
    const active = steps[steps.length - 1];
    if (active) active.bytes = count;
  };

  const ticking = setInterval(paint, 500);
  note.textContent = "Starting up";
  paint();

  backend
    .start(window.GLASSPRINT_PYTHON)
    .then(() => {
      clearInterval(ticking);
      splash.hidden = true;
      window.glassprintInit();
    })
    .catch((error) => {
      clearInterval(ticking);
      const last = steps[steps.length - 1];
      if (last && !last.until) last.until = Date.now();
      paint();
      // Python tracebacks arrive here in full. Neither end of one is the useful
      // part: the first line is always "Traceback (most recent call last):",
      // and the last is often a link to further reading. Prefer the line that
      // actually names the exception.
      const lines = String(error.message).split("\n").filter((line) => line.trim());
      const named = lines.filter((line) => /^[A-Za-z_.]*(Error|Exception)\b/.test(line.trim()));
      const blame = named[named.length - 1] || lines[lines.length - 1] || "unknown error";
      note.innerHTML = "<strong>Could not start.</strong><br />" + blame.slice(0, 200);
      stuck.hidden = false;
      stuck.innerHTML =
        "<span class='hint'>This page needs to be online the first time it runs, " +
        "and needs to be served over https rather than opened straight from a file.</span>";
    });
})();
"""

SPLASH = """
<div id="boot" class="boot">
  <div class="boot-card">
    <h1>glassprint</h1>
    <p id="boot-note">starting…</p>
    <ol id="boot-steps" class="boot-steps"></ol>
    <p id="boot-stuck" class="boot-stuck" hidden></p>
    <p class="hint">
      The first visit downloads Python and its imaging libraries — roughly
      50&nbsp;MB, most of it scipy. On a tablet that is a few minutes, and the
      last stretch is compiling rather than downloading, so the megabytes stop
      moving before it is finished. The browser keeps it all afterwards and
      later visits start in seconds.
    </p>
    <p class="hint">No image ever leaves the tablet.</p>
  </div>
</div>
"""

SPLASH_CSS = """
/* -- the browser build's first few seconds -- */
.boot {
  position: fixed;
  inset: 0;
  z-index: 50;
  display: grid;
  place-items: center;
  background: var(--bg, #14161a);
  padding: 24px;
}
.boot-card { max-width: 30rem; text-align: center; }
.boot-card h1 { font-size: 2rem; margin: 0 0 0.5rem; }
.boot-card p { margin: 0.5rem 0; }
.boot-steps {
  list-style: none;
  margin: 1rem auto;
  padding: 0;
  font-size: 0.85rem;
  text-align: left;
  display: inline-block;
  min-width: 18rem;
}
.boot-steps li { padding: 2px 0; color: var(--accent, #c9a227); }
.boot-steps li.done { color: var(--muted, #8b929c); }
.boot-steps li.done::before { content: "✓ "; }
.boot-steps li:not(.done)::before { content: "… "; }
.boot-stuck { color: var(--accent, #c9a227); }
"""


def python_sources() -> dict[str, str]:
    """Every module the bridge needs, keyed by filename."""
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(PACKAGE.glob("*.py"))
        if path.name not in SKIP
    }


def build() -> str:
    html = (WEB / "index.html").read_text(encoding="utf-8")
    style = (WEB / "style.css").read_text(encoding="utf-8")
    backend = (WEB / "backend.js").read_text(encoding="utf-8")
    app = (WEB / "app.js").read_text(encoding="utf-8")
    # Carried as a string, not a script tag: the page builds its worker from a
    # blob so that this stays one file with nothing beside it.
    worker = (WEB / "worker.js").read_text(encoding="utf-8")

    sources = python_sources()
    if "bridge.py" not in sources:
        raise SystemExit("bridge.py is missing — nothing would run")

    html = html.replace(
        '<link rel="stylesheet" href="/static/style.css" />',
        f"<style>\n{style}\n{SPLASH_CSS}</style>",
    )
    html = html.replace("<body>", "<body>" + SPLASH, 1)
    html = html.replace(
        '<script src="/static/backend.js"></script>\n<script src="/static/app.js"></script>',
        "\n".join(
            [
                "<script>",
                f"window.GLASSPRINT_PYTHON = {json.dumps(sources)};",
                f"window.GLASSPRINT_WORKER = {json.dumps(worker)};",
                "</script>",
                f"<script>\n{backend}\n</script>",
                f"<script>\n{app}\n</script>",
                f"<script>{BOOT}</script>",
            ]
        ),
    )
    if "/static/" in html:
        raise SystemExit("something still points at /static/ — it would 404 on a tablet")
    return html


def main() -> int:
    html = build()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"{OUTPUT.relative_to(ROOT)} — {len(html) / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
