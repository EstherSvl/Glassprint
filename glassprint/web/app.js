"use strict";

const $ = (id) => document.getElementById(id);

/* The page picks a backend before app.js runs: the local Python server on a
 * desktop, or Python inside this tab on a tablet. */
const backend = () => window.GlassprintBackend || window.GlassprintBackends.HttpBackend;

const state = {
  sessionId: null,
  images: {},
  view: "composite",
  target: "alpha",
  fadeMode: "none",
  lastPreview: null,
  inFlight: null,
  capabilities: null,
};

/* ------------------------------------------------------------------ status */

function setStatus(text, kind = "") {
  const el = $("status");
  el.textContent = text;
  el.className = "status" + (kind ? " " + kind : "");
}

/* ------------------------------------------------------------- reading form */

function num(id, fallback = null) {
  const raw = $(id).value.trim();
  if (raw === "") return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

function checkedValues(containerId) {
  return Array.from($(containerId).querySelectorAll("input:checked")).map((el) => el.value);
}

function buildSpec() {
  return {
    session_id: state.sessionId,
    keep: $("keep").value,
    tolerance: Number($("tolerance").value),
    use_claude: $("use-claude").checked,
    target: state.target,
    target_describe: $("target-describe").value,
    clip_to_shape: $("clip").checked,
    shape_grow: Number($("shape-grow").value),
    shape_feather: Number($("shape-feather").value),
    edge_feather: Number($("edge-feather").value),
    opacity: Number($("opacity").value),
    blend: $("blend").value,
    placement: {
      fit: $("fit").value,
      repeat_across: num("repeat-across"),
      repeat_mm: num("repeat-mm"),
      scale: Number($("scale").value),
      rotation: Number($("rotation").value),
      offset_x: Number($("offset-x").value),
      offset_y: Number($("offset-y").value),
      mirror: $("mirror").value,
      flip_h: $("flip-h").checked,
      flip_v: $("flip-v").checked,
    },
    color: {
      mode: $("color-mode").value,
      color: $("color-hex").value,
      color2: $("color2-hex").value,
      from_color: $("from-color-hex").value,
      strength: Number($("strength").value),
      hue_shift: Number($("hue-shift").value),
      saturation: Number($("saturation").value),
      brightness: Number($("brightness").value),
      contrast: Number($("contrast").value),
    },
    fade: {
      mode: state.fadeMode,
      what: $("fade-what").value,
      angle: Number($("fade-angle").value),
      center_x: Number($("fade-center-x").value),
      center_y: Number($("fade-center-y").value),
      start: Number($("fade-start").value),
      end: Number($("fade-end").value),
      curve: Number($("fade-curve").value),
      min_alpha: Number($("fade-min").value),
      max_alpha: Number($("fade-max").value),
      per_element: $("fade-per-element").checked,
      dissolve: Number($("fade-dissolve").value),
      seed: num("fade-seed", 0),
      layers: Number($("fade-layers").value),
      halftone_mm: Number($("fade-halftone").value),
      halftone_angle: Number($("fade-halftone-angle").value),
      invert: $("fade-invert").checked,
      cutoff: Number($("fade-cutoff").value),
    },
    glaze: {
      enabled: $("glaze-on").checked,
      glass: $("glass-color").value,
      palette: $("glaze-palette").value,
      colours: num("glaze-colours", 5),
      max_total: num("glaze-max-total", 5),
    },
    include_masks: state.view === "shape_mask" || state.view === "cutout_mask",
    // Without a white base the ink is a glaze, which is multiplicative — the
    // server renders it properly rather than us faking it with a blend mode.
    simulate: inkOnGlass()
      ? { glass: $("glass-color").value, layers: Math.max(1, Number($("fade-layers").value)) }
      : null,
  };
}

/* ---------------------------------------------------------------- previewing */

let previewTimer = null;

function schedulePreview(delay = 320) {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(runPreview, delay);
}

async function runPreview() {
  if (!state.images.base || !state.images.overlay) return;

  if (state.inFlight) state.inFlight.abort();
  const controller = new AbortController();
  state.inFlight = controller;
  setStatus("rendering…", "busy");

  try {
    const data = await backend().preview(buildSpec(), controller.signal);
    state.sessionId = data.session_id || state.sessionId;
    state.lastPreview = data;
    showView();
    renderReadout(data.summary);
    setStatus("ready");
  } catch (error) {
    if (error.name === "AbortError") return;
    setStatus(error.message, "error");
  } finally {
    if (state.inFlight === controller) state.inFlight = null;
  }
}

function showView() {
  const data = state.lastPreview;
  if (!data) return;
  // In no-white mode the glaze render replaces the plain composite.
  let key = state.view;
  if (state.view === "composite") {
    if (data.images.glazed) key = "glazed";
    else if (data.images.glaze) key = "glaze";
  }
  const src = data.images[key] || data.images.composite;
  const img = $("preview");
  img.src = src;
  img.hidden = false;
  $("empty-state").hidden = true;
}

function renderReadout(summary) {
  const pattern = summary.pattern || {};
  const [bw, bh] = summary.base_size || [0, 0];
  const [mmw, mmh] = summary.base_size_mm || [0, 0];
  const [x0, y0, x1, y1] = summary.shape_box || [0, 0, 0, 0];

  const dpi = summary.base_dpi[0] || 300;
  const toMm = (px) => Math.round((px / dpi) * 25.4 * 10) / 10;

  const rows = [
    ["Base", `${bw} × ${bh} px · ${mmw} × ${mmh} mm @ ${dpi} dpi`],
    [
      "Target area",
      `${x1 - x0} × ${y1 - y0} px · ${toMm(x1 - x0)} × ${toMm(y1 - y0)} mm · ` +
        `${Math.round(summary.shape_coverage * 100)}% of canvas`,
    ],
    ["Overlay reads as", `${pattern.is_pattern ? "a repeating pattern" : "a single motif"} — ${pattern.reason}`],
    ["Tiling", pattern.seamless ? "edges match, tiles cleanly" : "edges do not match, mirroring by default"],
    ["Suggested repeats", `${pattern.suggested_repeats} across the shape`],
    ["Cut-out", `${summary.plan} — ${Math.round(summary.cutout_coverage * 100)}% of the artwork kept`],
  ];

  // Below 10% a whole-number percentage rounds to a useless "0%".
  const asCoverage = (v) => (v < 0.1 ? `${(v * 100).toFixed(1)}%` : `${Math.round(v * 100)}%`);

  const fade = summary.fade || {};
  const warnings = [];
  if (fade.mode && fade.mode !== "none") {
    const faintest = fade.faintest_alpha || 0;
    const over = fade.elements ? ` over ${fade.elements} elements` : "";
    rows.push(["Fade", fade.describe + over]);
    rows.push(["Faintest ink", `${asCoverage(faintest)} coverage`]);

    // The dither floor is a white-underbase problem. Without white the ink is
    // a glaze, sparse coverage reads as a thinner tint rather than as specks,
    // and a plain tonal fade is fine.
    if (faintest > 0 && faintest < 0.12 && !inkOnGlass()) {
      warnings.push(
        `The thinnest ink is at ${asCoverage(faintest)} coverage. With a white underbase ` +
          "UV dithering tends to go speckly under about 12% — raise the end opacity, add " +
          "dissolve or a dot screen, or set a minimum printable ink level. Printing " +
          "without white? Switch the preview above and this stops mattering."
      );
    }
  }

  renderRecipes(summary.glaze);

  // Two kinds, and they were one kind for too long. A warning is something to
  // act on; a note is the tool saying which of several working paths it took.
  // Rendered identically, in the same alarm colour, a graceful fallback read as
  // a failure — and the one on a tablet recommended a fix a tablet cannot apply.
  const notes =
    warnings.map((n) => `<p class="warn-note">${escapeHtml(n)}</p>`).join("") +
    (summary.notes || []).map((n) => `<p class="note">${escapeHtml(n)}</p>`).join("");
  $("readout").innerHTML =
    "<dl>" +
    rows.map(([k, v]) => `<div><dt>${k}</dt><dd>${escapeHtml(String(v))}</dd></div>`).join("") +
    "</dl>" +
    notes;

  if (summary.plan_explanation) {
    $("plan-hint").textContent = summary.plan_explanation;
  }
}

function renderRecipes(glaze) {
  const box = $("glaze-recipes");
  if (!glaze) {
    box.innerHTML = "";
    return;
  }
  const rows = glaze.recipes
    .map((r) => {
      const warn = r.reachable ? "" : ' class="unreachable"';
      return (
        `<li${warn}><span class="swatch" style="background:${r.target}"></span>` +
        `<span class="arrow">&rarr;</span>` +
        `<span class="swatch" style="background:${r.achieved}"></span>` +
        `<span class="recipe">${escapeHtml(r.recipe)}</span></li>`
      );
    })
    .join("");
  box.innerHTML =
    `<p class="hint">${glaze.total_passes} passes: ${escapeHtml(glaze.stack.join(", "))}</p>` +
    `<ul class="recipes">${rows}</ul>` +
    (glaze.unreachable.length
      ? `<p class="note">${glaze.unreachable.length} colour(s) are brighter than the glass ` +
        "and print darker than asked — only a white base fixes that.</p>"
      : "");
}

function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/* ------------------------------------------------------------------ uploads */

/* Uploads run one after another. Dropping a base and an overlay together used
 * to start both at once, and with no session id yet each one opened its own —
 * so the preview then found only half its images. */
let uploadQueue = Promise.resolve();

function upload(file, role) {
  uploadQueue = uploadQueue.then(() => uploadOne(file, role));
  return uploadQueue;
}

async function uploadOne(file, role) {
  setStatus(`loading ${role}…`, "busy");
  try {
    const data = await backend().upload(file, role, state.sessionId);
    state.sessionId = data.session_id || state.sessionId;
    state.images[role] = data.image;
    describeImage(role, data.image);

    if (role === "base" && !$("basename").value) $("basename").value = data.image.name || "";
    if (state.images.base && state.images.overlay) {
      $("export-button").disabled = false;
      schedulePreview(0);
    } else {
      setStatus(`${role} loaded — add the ${role === "base" ? "overlay" : "base image"}`);
    }
  } catch (error) {
    setStatus(error.message, "error");
  }
}

function describeImage(role, image) {
  const thumb = $(`thumb-${role}`);
  thumb.src = image.thumb;
  thumb.hidden = false;
  $(`drop-${role}`).classList.add("loaded");

  const dpi = image.dpi_tagged ? `${image.dpi[0]} dpi` : `${image.dpi[0]} dpi (assumed)`;
  const alpha = image.has_alpha ? "transparent areas" : "no transparency";
  $(`meta-${role}`).textContent =
    `${image.width}×${image.height} px · ${image.size_mm[0]}×${image.size_mm[1]} mm · ${dpi} · ${alpha}`;
}

function wireDropZone(role) {
  const zone = $(`drop-${role}`);
  const input = $(`file-${role}`);

  input.addEventListener("change", () => {
    if (input.files && input.files[0]) upload(input.files[0], role);
  });
  ["dragenter", "dragover"].forEach((event) =>
    zone.addEventListener(event, (e) => {
      e.preventDefault();
      zone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((event) =>
    zone.addEventListener(event, (e) => {
      e.preventDefault();
      zone.classList.remove("over");
    })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) upload(file, role);
  });
}

/* ------------------------------------------------------------------- export */

async function runExport() {
  const button = $("export-button");
  button.disabled = true;
  setStatus("exporting…", "busy");

  const payload = buildSpec();
  payload.export = {
    formats: checkedValues("formats"),
    targets: checkedValues("targets"),
    include_base_format: $("include-base-format").checked,
    dpi: num("dpi"),
    width_mm: num("width-mm"),
    quality: 95,
    basename: $("basename").value,
    directory: $("export-dir").value,
  };

  try {
    const data = await backend().export(payload);

    const items = data.files
      .map((f) => {
        const detail =
          `${f.pixels[0]}×${f.pixels[1]} px @ ${f.dpi} dpi · ${f.size_mm[0]}×${f.size_mm[1]} mm` +
          `${f.alpha ? " · transparent" : ""}`;
        // Without a server there is no per-file link — the zip is the delivery.
        const label = f.download
          ? `<a href="${f.download}" download>${escapeHtml(f.file)}</a>`
          : escapeHtml(f.file);
        return f.pixels[0] ? `<li>${label} — ${detail}</li>` : `<li>${label}</li>`;
      })
      .join("");

    const heading = data.bundle
      ? `<p class="path"><a class="bundle" href="${data.bundle.download}" download="${escapeHtml(
          data.bundle.file
        )}">Save ${escapeHtml(data.bundle.file)}</a> — then unzip it in Files</p>`
      : `<p class="path">Written to ${escapeHtml(data.directory)}</p>`;
    $("export-result").innerHTML = heading + `<ul>${items}</ul>`;
    setStatus(`exported ${data.files.length} file${data.files.length === 1 ? "" : "s"}`);
  } catch (error) {
    $("export-result").innerHTML = "";
    setStatus(error.message, "error");
  } finally {
    button.disabled = false;
  }
}


/* -------------------------------------------------------------- calibration */

/* The profile outlives the tab. Recalibrating means printing another plate, so
 * losing it to a refresh would be a genuinely expensive mistake. */
const PROFILE_KEY = "glassprint.profile";

function saveProfile(profile) {
  try {
    window.localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  } catch (error) {
    /* Private browsing, or a full quota. The profile still works this session. */
  }
}

function storedProfile() {
  try {
    const raw = window.localStorage.getItem(PROFILE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (error) {
    return null;
  }
}

function showProfile(report) {
  const box = $("profile-summary");
  if (!report) {
    box.hidden = true;
    $("colour-tool").hidden = true;
    $("forget-profile").hidden = true;
    return;
  }

  const gamma = report.gamma.map((g) => g.toFixed(2)).join(" · ");
  const better = report.uncalibrated_error_levels / Math.max(report.error_levels, 0.1);
  const ground = { white: "white base", opaque: "opaque glass" }[report.substrate] || "glass";
  const rows = [
    [
      ground,
      `<span class="swatch" style="background:${report.glass}"></span>${report.glass}`,
    ],
    ["tone curve", `${gamma} <span class="hint">r · g · b</span>`],
    ["ink bleed", `${Math.round(report.muddiness * 100)}% outside its own channel`],
    ["accuracy", `${report.error_levels} levels — was ${report.uncalibrated_error_levels}`],
  ];
  // The repeated patch is the measurement measuring itself. Textured glass,
  // a shaky hand or a patchy lamp all show up here first, and every other
  // number is only meaningful against it — so it gets a verdict, not just a
  // figure.
  if (report.noise_levels != null) {
    const noise = report.noise_levels;
    const verdict =
      noise < 3 ? "clean" : noise < 8 ? "usable, but the fine distinctions are soft" : "too noisy to trust";
    rows.push([
      "repeatability",
      `${noise} levels between two prints of one colour — <b>${verdict}</b>`,
    ]);
  }
  // The two-column grid lives on .profile itself, so the terms and values go
  // straight in rather than inside a <dl> that would break the grid.
  box.innerHTML = rows.map(([term, value]) => `<dt>${term}</dt><dd>${value}</dd>`).join("");
  box.hidden = false;
  $("colour-tool").hidden = false;
  $("forget-profile").hidden = false;

  if (report.noise_levels != null && report.noise_levels >= 8) {
    setStatus(
      "the same colour printed twice came out " +
        `${report.noise_levels} levels apart — the surface or the light is louder than the ink`,
      "error"
    );
  } else if (better > 3) {
    setStatus(`calibrated — ${better.toFixed(0)}x closer than the assumption it replaces`);
  } else {
    setStatus("calibrated, but barely better than uncalibrated — check the photo", "error");
  }
}

async function getChart() {
  setStatus("drawing the chart…", "busy");
  try {
    const data = await backend().call("chart", {
      session_id: state.sessionId,
      substrate: $("chart-substrate").value,
    });
    state.sessionId = data.session_id || state.sessionId;

    const bytes = window.GlassprintBackends.base64ToBytes(data.data);
    const url = URL.createObjectURL(new Blob([bytes], { type: "image/png" }));
    $("chart-result").innerHTML =
      `<p class="path"><a class="bundle" href="${url}" download="${escapeHtml(
        data.file
      )}">Save ${escapeHtml(data.file)}</a> — ${data.size_mm[0]}×${data.size_mm[1]} mm, ${
        data.patches
      } patches</p><ol>${data.instructions
        .map((line) => `<li>${escapeHtml(line)}</li>`)
        .join("")}</ol>`;
    setStatus("chart ready to print");
  } catch (error) {
    setStatus(error.message, "error");
  }
}

async function readChart(file) {
  setStatus("reading the chart…", "busy");
  try {
    const buffer = await file.arrayBuffer();
    const data = await backend().call("calibrate", {
      session_id: state.sessionId,
      data: window.GlassprintBackends.bytesToBase64(new Uint8Array(buffer)),
      glass: $("glass-on").checked ? $("glass-color").value : "",
      substrate: $("chart-substrate").value,
    });
    state.sessionId = data.session_id || state.sessionId;
    saveProfile(data.profile);
    showProfile(data.report);
    updateColourAnswer();
    schedulePreview(0);
  } catch (error) {
    setStatus(error.message, "error");
  }
}

let colourTimer = null;

function updateColourAnswer(delay = 250) {
  clearTimeout(colourTimer);
  colourTimer = setTimeout(async () => {
    const wanted = $("want-hex").value.trim();
    if (!/^#[0-9a-fA-F]{6}$/.test(wanted)) return;
    try {
      const data = await backend().call("colour", {
        session_id: state.sessionId,
        wanted,
        glass: $("glass-on").checked ? $("glass-color").value : "",
      });
      const warn = data.note ? `<p class="warn">${escapeHtml(data.note)}</p>` : "";
      $("colour-answer").innerHTML =
        `<div class="ask"><span class="swatch" style="background:${data.ask_for}"></span>` +
        `ask the printer for <b>${data.ask_for}</b></div>${warn}`;
    } catch (error) {
      $("colour-answer").innerHTML = "";
    }
  }, delay);
}

async function restoreProfile() {
  const saved = storedProfile();
  if (!saved) return;
  try {
    const data = await backend().call("load_profile", {
      session_id: state.sessionId,
      profile: saved,
    });
    state.sessionId = data.session_id || state.sessionId;
    showProfile(data.report);
    updateColourAnswer(0);
  } catch (error) {
    // A profile from an older build is not worth an error message on startup.
    window.localStorage.removeItem(PROFILE_KEY);
  }
}

function wireCalibration() {
  $("chart-button").addEventListener("click", getChart);
  $("calibrate-button").addEventListener("click", () => $("file-chart").click());
  $("file-chart").addEventListener("change", () => {
    const file = $("file-chart").files[0];
    if (file) readChart(file);
    $("file-chart").value = "";
  });
  $("forget-profile").addEventListener("click", async () => {
    window.localStorage.removeItem(PROFILE_KEY);
    showProfile(null);
    try {
      await backend().call("load_profile", { session_id: state.sessionId, profile: null });
    } catch (error) {
      /* The bridge refuses a null profile; the page state is what matters. */
    }
    setStatus("calibration forgotten — previews are back to assuming");
    schedulePreview(0);
  });
  // Deliberately not linkColor(): that one also schedules a preview, and the
  // colour being asked about here has nothing to do with the artwork.
  $("want-colour").addEventListener("input", () => {
    $("want-hex").value = $("want-colour").value;
    updateColourAnswer();
  });
  $("want-hex").addEventListener("change", () => {
    const hex = $("want-hex").value.trim();
    if (/^#[0-9a-fA-F]{6}$/.test(hex)) $("want-colour").value = hex;
    updateColourAnswer(0);
  });
}

/* -------------------------------------------------------------------- wiring */


function bindSegmented(id, onChange) {
  $(id).addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    $(id).querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === button));
    onChange(button.dataset.value);
  });
}

function bindLive(ids) {
  ids.forEach((id) => {
    const el = $(id);
    const event = el.type === "range" || el.tagName === "SELECT" || el.type === "checkbox" ? "input" : "input";
    el.addEventListener(event, () => {
      updateLabels();
      schedulePreview();
    });
    if (el.type === "checkbox" || el.tagName === "SELECT") {
      el.addEventListener("change", () => {
        updateLabels();
        schedulePreview(0);
      });
    }
  });
}

function updateLabels() {
  const percent = (id) => `${Math.round(Number($(id).value) * 100)}%`;
  $("scale-value").textContent = percent("scale");
  $("opacity-value").textContent = percent("opacity");
  $("strength-value").textContent = percent("strength");
  $("saturation-value").textContent = percent("saturation");
  $("brightness-value").textContent = percent("brightness");
  $("contrast-value").textContent = percent("contrast");
  $("offset-x-value").textContent = percent("offset-x");
  $("offset-y-value").textContent = percent("offset-y");
  $("rotation-value").textContent = `${$("rotation").value}°`;
  $("hue-shift-value").textContent = `${$("hue-shift").value}°`;
  $("shape-feather-value").textContent = `${$("shape-feather").value} px`;
  $("shape-grow-value").textContent = `${$("shape-grow").value} px`;
  $("edge-feather-value").textContent = `${$("edge-feather").value} px`;

  const mode = $("color-mode").value;
  $("color-field").hidden = mode === "none" || mode === "mono";
  $("strength-field").hidden = mode === "none";
  $("color2-row").hidden = mode !== "duotone";
  $("from-color-row").hidden = mode !== "replace";
  $("color-label").textContent = mode === "duotone" ? "Highlight colour" : mode === "replace" ? "New colour" : "Colour";

  const tiling = $("fit").value === "tile" || $("fit").value === "auto";
  $("tile-controls").style.opacity = tiling ? "1" : "0.45";
  $("target-describe").hidden = state.target !== "describe";

  $("fade-angle-value").textContent = `${$("fade-angle").value}°`;
  $("fade-start-value").textContent = percent("fade-start");
  $("fade-end-value").textContent = percent("fade-end");
  $("fade-curve-value").textContent = Number($("fade-curve").value).toFixed(2);
  $("fade-dissolve-value").textContent = percent("fade-dissolve");
  $("fade-min-value").textContent = percent("fade-min");
  $("fade-max-value").textContent = percent("fade-max");
  $("fade-center-x-value").textContent = percent("fade-center-x");
  $("fade-center-y-value").textContent = percent("fade-center-y");
  const cutoff = Number($("fade-cutoff").value);
  $("fade-cutoff-value").textContent = cutoff > 0 ? `${Math.round(cutoff * 100)}%` : "off";

  const stack = Number($("fade-layers").value);
  $("fade-layers-value").textContent = stack > 0 ? `${stack} passes` : "off";

  const pitch = Number($("fade-halftone").value);
  $("fade-halftone-value").textContent = pitch > 0 ? `${pitch.toFixed(2)} mm` : "off";
  $("fade-halftone-angle-value").textContent = `${$("fade-halftone-angle").value}°`;

  $("glaze-body").hidden = !$("glaze-on").checked;
  $("fade-body").hidden = state.fadeMode === "none";
  $("fade-angle-field").hidden = state.fadeMode !== "linear";
  $("fade-center-row").hidden = state.fadeMode !== "radial";
  // Layers, screen and dissolve express the same ramp; the first set wins.
  $("fade-dissolve").closest(".field").style.opacity = pitch > 0 || stack > 0 ? "0.45" : "1";
  $("fade-halftone").closest(".row").style.opacity = stack > 0 ? "0.45" : "1";
}

function applyGlassBackdrop() {
  const noWhite = $("ink-mode").value === "none";
  // Without a white underbase the ink is a glaze, so it has to be multiplied
  // with the glass rather than laid on top of it — which means the glass
  // colour is no longer optional.
  if (noWhite) $("glass-on").checked = true;

  const on = $("glass-on").checked;
  const viewport = $("viewport");
  viewport.classList.toggle("on-glass", on);
  viewport.style.backgroundColor = on ? $("glass-color").value : "";
  $("glass-on").disabled = noWhite;
}

function inkOnGlass() {
  return $("ink-mode").value === "none" && $("glass-on").checked;
}

function linkColor(pickerId, hexId) {
  const picker = $(pickerId);
  const hex = $(hexId);
  picker.addEventListener("input", () => {
    hex.value = picker.value;
    schedulePreview();
  });
  hex.addEventListener("change", () => {
    if (/^#[0-9a-fA-F]{6}$/.test(hex.value.trim())) picker.value = hex.value.trim();
    schedulePreview(0);
  });
}

async function loadCapabilities() {
  try {
    const data = await backend().capabilities();
    state.capabilities = data;
    $("claude-row").hidden = !data.claude;

    // With no server there is nowhere on disk to write to, so the folder field
    // goes away and the export arrives as a download instead.
    const writesFiles = data.writes_files !== false;
    $("export-dir-row").hidden = !writesFiles;
    if (writesFiles) $("export-dir").value = data.default_export_dir || "";

    const bits = [];
    bits.push(data.semantic_selection ? "object selection on" : "colour/tone selection");
    if (data.subject_cutout) bits.push("subject cutout on");
    if (data.claude) bits.push("Claude available");
    if (!writesFiles) bits.push("running in this tab");
    $("tagline").textContent = `v${data.version} · ${bits.join(" · ")}`;
    setStatus("ready");
  } catch (error) {
    setStatus(
      backend().name === "http" ? "could not reach the local server" : error.message,
      "error"
    );
  }
}

function init() {
  wireDropZone("base");
  wireDropZone("overlay");

  bindSegmented("target-modes", (value) => {
    state.target = value;
    updateLabels();
    schedulePreview(0);
  });
  bindSegmented("view-modes", (value) => {
    state.view = value;
    const needsMasks = value === "shape_mask" || value === "cutout_mask";
    if (needsMasks && state.lastPreview && !state.lastPreview.images[value]) {
      schedulePreview(0);
    } else {
      showView();
    }
  });

  bindSegmented("fade-modes", (value) => {
    state.fadeMode = value;
    updateLabels();
    schedulePreview(0);
  });

  bindLive([
    "keep", "tolerance", "use-claude", "target-describe", "clip", "shape-grow", "shape-feather",
    "edge-feather", "fit", "mirror", "repeat-across", "repeat-mm", "scale", "rotation",
    "offset-x", "offset-y", "flip-h", "flip-v", "color-mode", "strength", "hue-shift",
    "saturation", "brightness", "contrast", "opacity", "blend",
    "fade-what", "fade-angle", "fade-start", "fade-end", "fade-curve", "fade-dissolve",
    "fade-min", "fade-max", "fade-center-x", "fade-center-y", "fade-cutoff",
    "fade-per-element", "fade-invert", "fade-seed", "fade-halftone", "fade-halftone-angle",
    "fade-layers", "glaze-on", "glaze-palette", "glaze-colours", "glaze-max-total",
  ]);

  $("glass-on").addEventListener("change", applyGlassBackdrop);
  $("ink-mode").addEventListener("change", () => {
    applyGlassBackdrop();
    schedulePreview(0);
  });
  $("glass-color").addEventListener("input", () => {
    $("glass-on").checked = true;
    applyGlassBackdrop();
    if (inkOnGlass()) schedulePreview();
    // The answer to "what should I ask for" depends on what it lands on.
    updateColourAnswer();
  });

  linkColor("color", "color-hex");
  linkColor("color2", "color2-hex");
  linkColor("from-color", "from-color-hex");

  $("refresh").addEventListener("click", () => schedulePreview(0));
  $("export-button").addEventListener("click", runExport);

  wireCalibration();

  updateLabels();
  applyGlassBackdrop();
  loadCapabilities().then(restoreProfile);
}

// The browser build has to finish starting Python before any of this is
// answerable, so it calls init() itself once that is done.
window.glassprintInit = init;
if (!window.GlassprintBackend) init();
