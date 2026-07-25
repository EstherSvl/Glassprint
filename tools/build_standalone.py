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

# cli.py wants typer and server.py wants fastapi; neither exists in the browser
# and nothing the page calls imports them.
SKIP = {"cli.py", "server.py"}

BOOT = """
/* Start Python, then hand the page over to app.js.
 *
 * This takes minutes on a tablet the first time, so the wait reports what it
 * is doing, how much has arrived and how long it has been going. Without a
 * clock that visibly moves there is no way to tell a slow connection from a
 * dead one, and the honest answer is usually "slow". */
(function () {
  const backend = window.GlassprintBackends.PyodideBackend;
  window.GlassprintBackend = backend;

  const splash = document.getElementById("boot");
  const note = document.getElementById("boot-note");

  const started = Date.now();
  let stage = "starting";
  let bytes = 0;

  function paint() {
    const seconds = Math.round((Date.now() - started) / 1000);
    const clock =
      seconds < 60 ? seconds + "s" : Math.floor(seconds / 60) + "m " + (seconds % 60) + "s";
    const arrived = bytes ? " · " + (bytes / 1048576).toFixed(1) + " MB" : "";
    note.textContent = stage + arrived + " · " + clock;
  }

  const ticking = setInterval(paint, 1000);
  backend.onProgress = (text) => {
    stage = text;
    paint();
  };
  // Bytes land in bursts; the once-a-second repaint is what shows them.
  backend.onBytes = (count) => {
    bytes = count;
  };
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
      // Python tracebacks arrive here in full. Neither end of one is the useful
      // part: the first line is always "Traceback (most recent call last):",
      // and the last is often a link to further reading. Prefer the line that
      // actually names the exception.
      const lines = String(error.message).split("\\n").filter((line) => line.trim());
      const named = lines.filter((line) => /^[A-Za-z_.]*(Error|Exception)\\b/.test(line.trim()));
      const blame = named[named.length - 1] || lines[lines.length - 1] || "unknown error";
      note.innerHTML =
        "<strong>Could not start.</strong><br />" +
        blame.slice(0, 200) +
        "<br /><span class='hint'>This page needs to be online the first time " +
        "it runs, and needs to be served over https rather than opened straight " +
        "from a file.</span>";
    });
})();
"""

SPLASH = """
<div id="boot" class="boot">
  <div class="boot-card">
    <h1>glassprint</h1>
    <p id="boot-note">starting…</p>
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
