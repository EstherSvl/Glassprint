"use strict";

/* Python, on a thread of its own.
 *
 * Pyodide is synchronous: importing scipy, or rendering a preview, is one long
 * call that cannot be interrupted. Run that on the page's own thread and the
 * page stops — the clock on the splash screen freezes mid-count, sliders stop
 * answering, and there is no way to tell slow work from a crash. Exactly when
 * reassurance matters most, the tab plays dead.
 *
 * Here the work is just as slow, but it happens somewhere else. The page keeps
 * painting, and "still going" stays visibly different from "stopped".
 */

/* Imported one at a time on purpose. Each of these takes seconds under
 * WebAssembly, and while one runs nothing else can report anything, so naming
 * the one in flight beforehand is the only progress there is to give. */
const LIBRARIES = [
  ["numpy", "starting numpy"],
  ["scipy.ndimage", "starting scipy — the slow one"],
  ["PIL.Image, PIL.ImageFilter", "starting Pillow"],
];

let handle = null;

function stage(text) {
  // Posted before the blocking call below it. Delivery does not depend on this
  // thread staying free, so the page hears about a step before it begins.
  self.postMessage({ type: "stage", text });
}

/* Count the bytes as the wheels arrive, so a slow download looks different
 * from a stuck one. Only wheels: they are read as array buffers, so handing
 * back a re-streamed body is harmless, where doing the same to the runtime's
 * own .wasm would break streaming compilation. If any part of this is not
 * supported, leave fetch alone — losing a counter beats breaking a download.
 */
function trackWheelDownloads(report) {
  const original = self.fetch;
  try {
    new Response(new ReadableStream());
  } catch {
    return () => {};
  }

  let total = 0;
  self.fetch = async (input, init) => {
    const response = await original(input, init);
    const url = typeof input === "string" ? input : (input && input.url) || "";
    if (!url.endsWith(".whl") || !response.body) return response;

    try {
      const reader = response.body.getReader();
      const counted = new ReadableStream({
        async pull(controller) {
          const { done, value } = await reader.read();
          if (done) {
            controller.close();
            return;
          }
          total += value.byteLength;
          report(total);
          controller.enqueue(value);
        },
        cancel(reason) {
          return reader.cancel(reason);
        },
      });
      return new Response(counted, { headers: response.headers, status: response.status });
    } catch {
      return response;
    }
  };

  return () => {
    self.fetch = original;
  };
}

async function boot(base, sources) {
  stage("fetching Python");
  // A module import rather than importScripts: Pyodide ships as an ES module
  // and refuses to run in a classic worker.
  const { loadPyodide } = await import(base + "pyodide.mjs");
  const pyodide = await loadPyodide({ indexURL: base });

  stage("fetching numpy, scipy and Pillow");
  const untrack = trackWheelDownloads((total) => self.postMessage({ type: "bytes", total }));
  try {
    await pyodide.loadPackage(["numpy", "scipy", "pillow"], {
      messageCallback: (text) => {
        // "Loading …" as they start, "Loaded …" once downloaded and installing.
        if (/^Loaded/.test(text)) stage("unpacking the libraries");
      },
    });
  } finally {
    untrack();
  }

  for (const [module, label] of LIBRARIES) {
    stage(label);
    try {
      pyodide.runPython("import " + module);
    } catch {
      // Whichever one failed, the cause and the cure are the same.
      throw new Error("the imaging libraries did not load — check the connection and reload");
    }
  }

  stage("unpacking glassprint");
  const root = "/lib/glassprint-src";
  pyodide.FS.mkdirTree(root + "/glassprint");
  for (const [name, source] of Object.entries(sources)) {
    pyodide.FS.writeFile(root + "/glassprint/" + name, source);
  }
  // Ahead of site-packages, so a stale copy can never win.
  pyodide.runPython("import sys; sys.path.insert(0, " + JSON.stringify(root) + ")");

  stage("starting glassprint");
  handle = pyodide.runPython("from glassprint.bridge import handle; handle");
}

self.onmessage = async (event) => {
  const message = event.data || {};

  if (message.type === "boot") {
    try {
      await boot(message.pyodideUrl, message.sources);
      self.postMessage({ type: "ready" });
    } catch (error) {
      self.postMessage({ type: "error", message: String((error && error.message) || error) });
    }
    return;
  }

  if (message.type === "call") {
    // Everything crosses as a JSON string, the same shape the HTTP server
    // speaks, so neither side needs to know which one it is talking to.
    let json;
    try {
      json = handle
        ? handle(message.method, JSON.stringify(message.payload || {}))
        : JSON.stringify({ error: "Python is still starting up." });
    } catch (error) {
      json = JSON.stringify({ error: String((error && error.message) || error) });
    }
    self.postMessage({ type: "result", id: message.id, json });
  }
};
