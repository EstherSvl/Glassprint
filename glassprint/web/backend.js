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

  async drop(role, sessionId) {
    return this.call("drop", { role, session_id: sessionId });
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

  async call(method, payload) {
    const response = await fetch(`/api/call/${method}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload || {}),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `${method} failed`);
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
  worker: null,
  ready: false,
  onProgress: () => {},
  onBytes: () => {},
  pending: new Map(),
  nextId: 1,

  /* Pyodide is tens of megabytes, so this is deliberately explicit rather than
   * hidden behind the first preview. It runs in a worker: see web/worker.js
   * for why. */
  start(sources) {
    if (this.ready) return Promise.resolve();

    const source = window.GLASSPRINT_WORKER;
    if (!source) return Promise.reject(new Error("this build is missing its worker script"));

    // Built from a blob so the single-file page stays a single file, and a
    // module worker because Pyodide ships as an ES module.
    const worker = new Worker(
      URL.createObjectURL(new Blob([source], { type: "text/javascript" })),
      { type: "module" }
    );
    this.worker = worker;

    const started = new Promise((resolve, reject) => {
      worker.onmessage = (event) => {
        const message = event.data || {};
        if (message.type === "stage") {
          this.onProgress(message.text);
        } else if (message.type === "bytes") {
          this.onBytes(message.total);
        } else if (message.type === "ready") {
          this.ready = true;
          resolve();
        } else if (message.type === "error") {
          reject(new Error(message.message));
        } else if (message.type === "result") {
          const settle = this.pending.get(message.id);
          if (settle) {
            this.pending.delete(message.id);
            settle(message.json);
          }
        }
      };
      worker.onerror = (event) =>
        reject(new Error(event.message || "the worker could not start"));
    });

    worker.postMessage({
      type: "boot",
      pyodideUrl: window.GLASSPRINT_PYODIDE_URL || PYODIDE_URL,
      sources,
    });
    return started;
  },

  call(method, payload) {
    return new Promise((resolve, reject) => {
      if (!this.ready) {
        reject(new Error("Python is still starting up."));
        return;
      }
      const id = this.nextId++;
      this.pending.set(id, (json) => {
        const data = JSON.parse(json);
        if (data.error) reject(new Error(data.error));
        else resolve(data.ok);
      });
      this.worker.postMessage({ type: "call", id, method, payload: payload || {} });
    });
  },

  async capabilities() {
    return { ...(await this.call("capabilities")), writes_files: false };
  },

  async upload(file, role) {
    const buffer = await file.arrayBuffer();
    return this.call("upload", {
      role,
      filename: file.name,
      data: bytesToBase64(new Uint8Array(buffer)),
    });
  },

  async drop(role) {
    return this.call("drop", { role });
  },

  async preview(spec, signal) {
    // The worker renders one at a time. A superseded preview cannot be pulled
    // back, but its result can be dropped rather than painted over a newer one.
    if (signal && signal.aborted) throw abortError();
    const data = await this.call("preview", spec);
    if (signal && signal.aborted) throw abortError();
    return data;
  },

  async export(payload) {
    // No filesystem to write to: everything comes back as one zip the browser
    // hands to the Files app.
    const request = { ...payload, export: { ...payload.export, bundle: true } };
    const data = await this.call("export", request);
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
