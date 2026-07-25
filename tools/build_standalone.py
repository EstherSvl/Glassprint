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
 * The runtime is a few tens of megabytes on first visit and cached by the
 * browser afterwards, so the wait is announced rather than hidden. */
(function () {
  const backend = window.GlassprintBackends.PyodideBackend;
  window.GlassprintBackend = backend;

  const splash = document.getElementById("boot");
  const note = document.getElementById("boot-note");
  backend.onProgress = (text) => {
    note.textContent = text;
  };

  backend
    .start(window.GLASSPRINT_PYTHON)
    .then(() => {
      splash.hidden = true;
      window.glassprintInit();
    })
    .catch((error) => {
      // Python tracebacks arrive here in full. The line that names the problem
      // is the last one, not the first — "Traceback (most recent call last):"
      // tells nobody anything.
      const lines = String(error.message).split("\\n").filter((line) => line.trim());
      note.innerHTML =
        "<strong>Could not start.</strong><br />" +
        (lines[lines.length - 1] || "unknown error").slice(0, 200) +
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
      The first visit downloads Python and its imaging libraries — around 40&nbsp;MB,
      once. After that this page works with no network at all, and no image ever
      leaves the tablet.
    </p>
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
