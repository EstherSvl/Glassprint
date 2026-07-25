"use strict";

/* Where the work happens.
 *
 * Two of these exist. `HttpBackend` talks to the local Python server — that is
 * the desktop tool. `PyodideBackend` runs the same Python inside the browser
 * tab, which is how this works on an iPad, where there is no terminal to start
 * a server from and no way to install anything.
 *
 * Both return the same shapes, so app.js never learns which one it has.
 */

/* ---------------------------------------------------------------- over HTTP */

const HttpBackend = {
  name: "http",

  async capabilities() {
    const response = await fetch("/api/capabilities");
    return response.json();
  },

  async upload(file, role, sessionId) {
    const form = new FormData();
    form.append("file", file);
    form.append("role", role);
    if (sessionId) form.append("session_id", sessionId);

    const response = await fetch("/api/upload", { method: "POST", body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "upload failed");
    return data;
  },

  async preview(spec, signal) {
    const response = await fetch("/api/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
      signal,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "preview failed");
    return data;
  },

  async export(payload) {
    const response = await fetch("/api/export", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "export failed");
    return data;
  },
};

/* ------------------------------------------------------- inside the browser */

/* Where the Python runtime comes from. Overridable so a page can point at a
 * copy it hosts itself rather than the public CDN. */
const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";

const PyodideBackend = {
  name: "pyodide",
  pyodide: null,
  onProgress: () => {},

  /* Pyodide is tens of megabytes, so this is deliberately explicit rather than
   * hidden behind the first preview. The caller reports progress to the page. */
  async start(sources) {
    if (this.handle) return this.pyodide;

    this.onProgress("fetching Python…");
    const base = window.GLASSPRINT_PYODIDE_URL || PYODIDE_URL;
    if (!window.loadPyodide) await loadScript(base + "pyodide.js");
    // Published before the rest of the setup, so a retry after a failed step
    // reuses the runtime instead of downloading it again.
    const pyodide = this.pyodide || (await loadPyodide({ indexURL: base }));
    this.pyodide = pyodide;

    this.onProgress("fetching numpy, scipy and Pillow…");
    await pyodide.loadPackage(["numpy", "scipy", "pillow"]);
    // loadPackage reports a failed download to the console and carries on, so
    // check for ourselves — otherwise the first symptom is an unreadable
    // traceback several steps later.
    //
    // Ask Python whether the imports work rather than inspecting
    // loadedPackages: that is keyed by each package's display name, which is
    // "Pillow" where the request was "pillow", so comparing names invents
    // failures that did not happen.
    try {
      pyodide.runPython("import numpy, scipy.ndimage, PIL.Image, PIL.ImageFilter");
    } catch (error) {
      throw new Error("the imaging libraries did not load — check the connection and reload");
    }

    this.onProgress("unpacking glassprint…");
    const root = "/lib/glassprint-src";
    pyodide.FS.mkdirTree(root + "/glassprint");
    for (const [name, source] of Object.entries(sources)) {
      pyodide.FS.writeFile(`${root}/glassprint/${name}`, source);
    }
    // Ahead of site-packages, so a stale copy can never win.
    pyodide.runPython(`import sys; sys.path.insert(0, ${JSON.stringify(root)})`);

    this.onProgress("starting up…");
    this.handle = pyodide.runPython("from glassprint.bridge import handle; handle");
    return pyodide;
  },

  /* Every call crosses as a JSON string in both directions. Pyodide can pass
   * richer objects, but they need explicit destruction to avoid leaking, and
   * text keeps this backend interchangeable with the HTTP one. */
  call(method, payload) {
    if (!this.handle) throw new Error("Python is still starting up.");
    const raw = this.handle(method, JSON.stringify(payload || {}));
    const data = JSON.parse(raw);
    if (data.error) throw new Error(data.error);
    return data.ok;
  },

  async capabilities() {
    return { ...this.call("capabilities"), writes_files: false };
  },

  async upload(file, role) {
    const buffer = await file.arrayBuffer();
    return this.call("upload", {
      role,
      filename: file.name,
      data: bytesToBase64(new Uint8Array(buffer)),
    });
  },

  async preview(spec, signal) {
    // Python here is synchronous and holds the one thread the page has, so a
    // superseded preview cannot be cancelled mid-render — only skipped.
    if (signal && signal.aborted) throw abortError();
    await new Promise((resolve) => setTimeout(resolve, 0));
    if (signal && signal.aborted) throw abortError();
    return this.call("preview", spec);
  },

  async export(payload) {
    // No filesystem to write to: everything comes back as one zip the browser
    // hands to the Files app.
    const request = { ...payload, export: { ...payload.export, bundle: true } };
    const data = this.call("export", request);
    const zip = data.bundle;
    const blob = new Blob([base64ToBytes(zip.data)], { type: "application/zip" });
    return {
      ...data,
      bundle: { file: zip.file, download: URL.createObjectURL(blob) },
      files: data.files,
    };
  },
};

/* ------------------------------------------------------------------ helpers */

function abortError() {
  const error = new Error("superseded");
  error.name = "AbortError";
  return error;
}

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const tag = document.createElement("script");
    tag.src = src;
    tag.onload = resolve;
    tag.onerror = () => reject(new Error(`could not load ${src}`));
    document.head.appendChild(tag);
  });
}

function bytesToBase64(bytes) {
  // btoa wants a string, and String.fromCharCode blows the stack on a whole
  // image at once, so feed it in chunks.
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

function base64ToBytes(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

window.GlassprintBackends = { HttpBackend, PyodideBackend, bytesToBase64, base64ToBytes };
