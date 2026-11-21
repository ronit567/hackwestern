"use strict";
// Renders only what is in reports/web/*.json (contract: PLAN.md "Results frontend"). Nothing is computed from guesses.

const q = new URLSearchParams(location.search);
// ?data=<dir/> reads another export dir (used to check the metrics sections against a fixture); ?theme=light|dark forces a theme
const DATA = q.get("data") || "../reports/web/";
if (q.get("theme")) document.documentElement.dataset.theme = q.get("theme");

const RUN = {
  export: "uv run python -m orew.export",
  evaluate: "uv run python -m orew.evaluate --split val",
};
const SIGNALS = {
  map: "Arterial MAP", sbp: "Arterial systolic", dbp: "Arterial diastolic", nibp_map: "Cuff MAP (NIBP)",
  hr: "Heart rate", spo2: "SpO2", etco2: "End-tidal CO2", rr: "Respiratory rate", fio2: "FiO2",
  tv: "Tidal volume", peep: "PEEP", pip: "Peak airway pressure", sevo: "Sevoflurane", des: "Desflurane",
  mac: "MAC", ppf_rate: "Propofol infusion", rftn_rate: "Remifentanil infusion",
  phen_rate: "Phenylephrine infusion", nepi_rate: "Norepinephrine infusion", bis: "BIS",
  bt: "Body temperature", beat_pp: "Pulse pressure, per beat", beat_dpdt: "Max dP/dt, per beat",
  beat_sys_area: "Systolic area, per beat", beat_hr: "Beat rate from arterial line",
  beat_ppv: "Pulse-pressure variation", beat_sbp_sd: "Systolic variability",
  pleth_amp: "Pleth amplitude", pleth_cv: "Pleth stability",
};

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null && v !== false) el.setAttribute(k, v);
  el.append(...kids.flat(9).filter((k) => k != null && k !== false));
  return el;
}
function s(tag, attrs, ...kids) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs || {})) if (v != null && v !== false) el.setAttribute(k, v);
  el.append(...kids.flat(9).filter((k) => k != null && k !== false));
  return el;
}
const fill = (id, ...nodes) => document.querySelector(`#${id} .body`).replaceChildren(...nodes);
const empty = (what, cmd) => h("p", { class: "empty" }, `${what} Run `, h("code", {}, cmd), " to fill it.");
const fmt = (v, d = 0) => (v == null ? "n/a" : v.toLocaleString("en", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v, d = 0) => (v == null ? "n/a" : `${fmt(v * 100, d)} %`);
const counts = (o, name = (k) => k, sortByKey) =>
  Object.entries(o).sort(sortByKey ? (a, b) => a[0].localeCompare(b[0]) : (a, b) => b[1] - a[1])
    .map(([k, n]) => `${name(k)}: ${n}`).join(", ");

async function get(path) {
  try {
    const r = await fetch(DATA + path);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

function overview(c) {
  if (!c) return fill("overview", empty("No cohort summary yet.", RUN.export));
  const facts = [
    ["Surgeries downloaded", `${fmt(c.n_cached)} of ${fmt(c.n_cohort)} in the cohort`],
    ["Patients", fmt(c.n_subjects)],
    ["Age, median (IQR)", `${fmt(c.age.median)} (${fmt(c.age.q1)}–${fmt(c.age.q3)}) years`],
    ["Sex", counts(c.sex, (k) => ({ F: "Female", M: "Male" })[k] || k)],
    ["ASA class", counts(c.asa, (k) => `ASA ${k}`, true)],
    ["Emergency", `${fmt(c.emergency.n)} of ${fmt(c.emergency.of)}`],
    ["Median duration", `${fmt(c.median_duration_min)} min`],
    ["Department", counts(c.department)],
  ];
  if (c.events) for (const [ev, e] of Object.entries(c.events))
    facts.push([`${cap(ev)} events`, `${fmt(e.n_events)} in ${fmt(e.n_cases)} surgeries`]);
  if (c.splits) facts.push(["Split (patients)", ["train", "val", "test"].map((k) => `${k} ${fmt(c.splits[k]?.n_subjects)}`).join(", ")]);

  const rows = c.signals.map((g) =>
    h("tr", {},
      h("th", { scope: "row" }, SIGNALS[g.name] || g.name),
      h("td", { class: "bar", "aria-hidden": "true" }, h("span", { style: `width:${g.pct_time}%` })),
      h("td", {}, `${fmt(g.pct_time)} %`),
      h("td", {}, `${fmt(g.pct_cases)} %`)));
  fill("overview",
    h("dl", { class: "facts" }, facts.map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, v)))),
    h("h3", {}, "Signal availability"),
    h("p", { class: "note" }, "Share of 2 s samples with a value, averaged over downloaded surgeries, and share of surgeries with any value."),
    h("div", { class: "scroll" }, h("table", { class: "signals" },
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Signal"), h("td", {}),
        h("th", { scope: "col" }, "Of time"), h("th", { scope: "col" }, "Of surgeries"))),
      h("tbody", {}, rows))));
}
const cap = (t) => t[0].toUpperCase() + t.slice(1);

// ---- chart helpers
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
function ticks(lo, hi, n = 4) {
  const raw = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((st) => st >= raw);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6));
  return out;
}
function pathD(xs, ys, x, y) {
  let d = "", pen = false;
  for (let i = 0; i < xs.length; i++) {
    if (ys[i] == null) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${x(xs[i]).toFixed(1)},${y(ys[i]).toFixed(1)}`;
    pen = true;
  }
  return d;
}
const finite = (a) => a.filter((v) => v != null);
const median = (a) => { const b = [...a].sort((p, q) => p - q); return b.length ? b[b.length >> 1] : null; };

// ---- case explorer
const PANELS = [
  { key: "map", label: "Arterial MAP (line) and cuff MAP (circles), mmHg", unit: "mmHg", clamp: [20, 200], thr: 65, h: 150, event: "hypotension" },
  { key: "spo2", label: "SpO2, %", unit: "%", clamp: [50, 100], thr: 90, h: 100, event: "hypoxemia" },
  { key: "hr", label: "Heart rate, bpm", unit: "bpm", clamp: [20, 220], h: 90 },
  { key: "etco2", label: "End-tidal CO2, mmHg", unit: "mmHg", clamp: [0, 100], h: 80 },
  { key: "bis", label: "BIS", unit: "", clamp: [0, 100], h: 80 },
  { key: "ppf_rate", label: "Propofol infusion, mL/h", unit: "mL/h", clamp: [0, 1000], h: 60, cls: "drug", zero: true },
  { key: "rftn_rate", label: "Remifentanil infusion, mL/h", unit: "mL/h", clamp: [0, 1000], h: 60, cls: "drug", zero: true },
];
const PHASES = [["Induction", "anestart", "opstart"], ["Surgery", "opstart", "opend"], ["Emergence", "opend", "aneend"]];
const RISK_H = 80;
let caseId, caseData, picker;

function caseShell() {
  picker = h("select", { id: "case-pick" }, index.map((c) =>
    h("option", { value: c.caseid }, `${c.caseid}: ${c.opname || "operation not recorded"}, ${c.department || "department not recorded"}, ${fmt(c.duration_min)} min`)));
  picker.onchange = () => { caseId = +picker.value; drawCase(); };
  caseId = index[0].caseid;
  fill("cases",
    h("p", {}, `${index.length} surgeries exported. Each is shown whole, from case start; values are the raw monitor numbers before artifact cleaning, in 10 s steps.`),
    h("div", { class: "pick" }, h("label", { for: "case-pick" }, "Surgery"), picker),
    h("div", { id: "case-view" }));
}

async function drawCase() {
  const id = caseId, view = document.getElementById("case-view");
  if (caseData?.caseid !== id) {
    const c = await get(`cases/${id}.json`);
    if (id !== caseId) return; // another surgery was picked while this one loaded
    caseData = c;
  }
  const c = caseData;
  if (!c) return view.replaceChildren(empty(`Surgery ${id} is listed but its file is missing.`, RUN.export));
  const d = c.demographics;
  const who = [
    { F: "Female", M: "Male" }[d.sex] || "Sex not recorded", d.age != null && `${fmt(d.age)} years`,
    d.bmi != null && `BMI ${fmt(d.bmi, 1)}`, d.asa != null && `ASA ${fmt(d.asa)}`, d.emop ? "emergency" : "planned",
  ].filter(Boolean).join(", ");
  const notes = [];
  if (!c.events) notes.push(empty("Events are not labelled yet.", "uv run python -m orew.labels && " + RUN.export));
  if (!c.risk) notes.push(empty("No risk curves yet: they need the trained model.", "uv run python -m orew.explain && " + RUN.export));
  else if (c.split === "train") notes.push(h("p", { class: "note" }, "This is a training surgery, so its risk curves are out-of-fold predictions."));
  const readout = h("p", { class: "readout", "aria-live": "polite" }, "Point at the chart, or focus it and use the arrow keys, to read values.");
  view.replaceChildren(
    h("p", { class: "who" }, `${who}. ${d.department || ""}${d.opname ? `: ${d.opname}` : ""}${d.optype ? ` (${d.optype})` : ""}.`),
    ...notes, readout, caseChart(c, view.clientWidth, readout), alertList(c));
}

function caseChart(c, W, readout) {
  const ML = 40, MR = 12, TOP = 30, GAP = 26;
  const T = c.t_min, tMax = Math.max(c.duration_min || 0, T[T.length - 1] || 1);
  const x = lin(0, tMax, ML, W - MR);
  const panels = PANELS.filter((p) => finite(c.series[p.key]).length);
  const risks = Object.entries(c.risk || {});
  let yTop = TOP;
  const rows = [...panels.map((p) => ({ ...p, vals: c.series[p.key] })),
    ...risks.map(([ev, r]) => ({ key: `risk-${ev}`, label: `${cap(ev)} risk in the next 10 min, % (dashed: alert threshold; triangles: alerts)`, unit: "%", clamp: [0, 1], h: RISK_H, cls: "risk", zero: true, risk: r, ev }))]
    .map((p) => { const r = { ...p, y0: yTop + 18 }; yTop = r.y0 + p.h + GAP; return r; });
  const H = yTop + 6;
  const tStep = [10, 30, 60, 120].find((st) => x(st) - x(0) >= 44) || 120;
  const g = [];

  // anesthesia-record grid: faint 5 min columns, stronger every tStep, across every panel
  for (const r of rows)
    for (let t = 0; t <= tMax; t += 5)
      g.push(s("line", { class: t % tStep ? "grid" : "grid-major", x1: x(t), x2: x(t), y1: r.y0, y2: r.y0 + r.h }));
  for (let t = 0; t <= tMax; t += tStep) g.push(s("text", { x: x(t), y: H - 6, "text-anchor": "middle" }, t ? `${t}` : "0 min"));

  // phase strip + marker lines
  const ph = c.phases_min;
  for (const [name, a, b] of PHASES) {
    if (ph[a] == null || ph[b] == null) continue;
    const w = x(ph[b]) - x(ph[a]);
    g.push(s("rect", { class: `phase-${a}`, x: x(ph[a]), y: 4, width: Math.max(w, 1), height: 16, rx: 2 },
      s("title", {}, `${name}: ${fmt(ph[a], 1)}–${fmt(ph[b], 1)} min`)));
    if (w > name.length * 7 + 8) g.push(s("text", { class: "phase-label", x: x(ph[a]) + w / 2, y: 16, "text-anchor": "middle" }, name));
  }
  for (const k of ["anestart", "opstart", "opend", "aneend"])
    if (ph[k] != null) g.push(s("line", { class: "phase", x1: x(ph[k]), x2: x(ph[k]), y1: 20, y2: H - 20 }));

  const summary = [];
  for (const r of rows) {
    const vals = r.risk ? r.risk.p10 : r.vals, ts = r.risk ? r.risk.t_min : T;
    const seen = finite(vals).filter((v) => v >= r.clamp[0] && v <= r.clamp[1]);
    let lo = r.zero ? 0 : Math.min(...seen, r.thr ?? Infinity), hi = Math.max(...seen, r.thr ?? -Infinity, r.zero ? 1 : -Infinity);
    if (r.risk) [lo, hi] = [0, 1];
    const tk = ticks(lo, hi, r.h > 90 ? 4 : 2);
    lo = Math.min(lo, tk[0]); hi = Math.max(hi, tk[tk.length - 1]);
    const y = lin(lo, hi, r.y0 + r.h, r.y0);
    g.push(s("text", { class: "title", x: ML, y: r.y0 - 6 }, r.label));
    for (const v of tk) {
      g.push(s("line", { class: "grid", x1: ML, x2: W - MR, y1: y(v), y2: y(v) }));
      g.push(s("text", { x: ML - 6, y: y(v) + 4, "text-anchor": "end" }, r.risk ? `${v * 100}` : `${v}`));
    }
    const clip = `clip-${r.key}`;
    g.push(s("clipPath", { id: clip }, s("rect", { x: ML, y: r.y0, width: W - MR - ML, height: r.h })));
    const inner = [];
    for (const e of (c.events || []).filter((e) => e.type === (r.event || r.ev)))
      inner.push(s("rect", { class: `ev ev-${e.type}`, x: x(e.onset_min), y: r.y0, width: Math.max(x(e.end_min) - x(e.onset_min), 2), height: r.h },
        s("title", {}, `${cap(e.type)} ${fmt(e.onset_min, 1)}–${fmt(e.end_min, 1)} min`)));
    if (r.thr != null) inner.push(s("line", { class: "thr", x1: ML, x2: W - MR, y1: y(r.thr), y2: y(r.thr) }));
    if (r.risk?.threshold != null) inner.push(s("line", { class: "thr", x1: ML, x2: W - MR, y1: y(r.risk.threshold), y2: y(r.risk.threshold) }));
    inner.push(s("path", { class: `line s-${r.cls || r.key}`, d: pathD(ts, vals, x, y) }));
    if (r.key === "map") // a cuff reading is held for up to 30 s on the grid, so draw only where the value changes
      c.series.nibp_map.forEach((v, i, a) => v != null && v !== a[i - 1] && inner.push(s("circle", { class: "nibp", cx: x(T[i]), cy: y(v), r: 3 })));
    for (const a of (c.alerts || []).filter((a) => r.risk && a.type === r.ev))
      inner.push(s("path", { class: "alert", d: `M${x(a.t_min)},${r.y0 + 9}l-5,-9h10z` }, s("title", {}, `Alert at ${fmt(a.t_min, 1)} min`)));
    g.push(s("g", { "clip-path": `url(#${clip})` }, inner));
    summary.push(`${r.label}: ${seen.length ? `median ${fmt(median(seen) * (r.risk ? 100 : 1))}, range ${fmt(Math.min(...seen) * (r.risk ? 100 : 1))} to ${fmt(Math.max(...seen) * (r.risk ? 100 : 1))}` : "no data"}.`);
  }

  // crosshair readout, by pointer or arrow keys
  const cursor = s("line", { class: "cursor", y1: 20, y2: H - 20, visibility: "hidden" });
  const svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, tabindex: 0, role: "img",
    "aria-label": `Vital signs over ${fmt(tMax)} minutes of surgery ${c.caseid}. ${summary.join(" ")} Use the left and right arrow keys to read values.` },
  g, cursor);
  const hit = s("rect", { x: ML, y: 0, width: W - MR - ML, height: H, fill: "transparent" });
  svg.append(hit);
  let at = -1;
  const show = (i) => {
    at = Math.max(0, Math.min(T.length - 1, i));
    const t = T[at], parts = [`${fmt(t, 1)} min`];
    for (const p of panels) {
      const v = c.series[p.key][at];
      parts.push(`${SIGNALS[p.key] || p.label} ${v == null ? "no value" : `${fmt(v)} ${p.unit}`.trim()}`);
      if (p.key === "map" && c.series.nibp_map[at] != null) parts.push(`cuff MAP ${fmt(c.series.nibp_map[at])} mmHg`);
    }
    for (const [ev, r] of risks) {
      let j = r.t_min.findLastIndex((u) => u <= t);
      if (j >= 0 && r.p10[j] != null) parts.push(`${ev} 10 min risk ${pct(r.p10[j])}`);
    }
    readout.textContent = parts.join(", ");
    cursor.setAttribute("x1", x(t)); cursor.setAttribute("x2", x(t)); cursor.setAttribute("visibility", "visible");
  };
  const nearest = (t) => { let i = T.findIndex((u) => u >= t); return i < 0 ? T.length - 1 : i; };
  const toT = (px) => ((px - ML) / (W - MR - ML)) * tMax;
  hit.addEventListener("pointermove", (e) => show(nearest(toT(e.clientX - svg.getBoundingClientRect().left))));
  svg.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 6, ArrowLeft: -6, PageDown: 60, PageUp: -60 }[e.key]; // 6 bins = 1 min
    if (step) { e.preventDefault(); show((at < 0 ? 0 : at) + (e.shiftKey ? step * 10 : step)); }
    if (e.key === "Home" || e.key === "End") { e.preventDefault(); show(e.key === "Home" ? 0 : T.length - 1); }
  });
  return h("div", { class: "chart" }, svg, h("ul", { class: "vh" }, summary.map((t) => h("li", {}, t))));
}

function alertList(c) {
  if (!c.alerts?.length) return null;
  return h("div", {}, h("h3", {}, "Alerts in this surgery"), h("ul", { class: "alerts" }, c.alerts.map((a) =>
    h("li", {}, h("strong", {}, `${fmt(a.t_min, 1)} min, ${a.type}`), `: 10 min risk ${pct(a.p10)}, led by the ${a.lead_specialist} model. `,
      `Drivers: ${(a.top_drivers || []).join("; ")}.`))));
}

let cohort, metrics, index;

function renderAll() {
  overview(cohort);
  for (const id of ["headline", "metrics", "calibration", "alerts", "subgroups"])
    if (!metrics) fill(id, empty("Not computed yet.", RUN.evaluate));
  if (!index?.length) fill("cases", empty("No surgeries exported yet.", RUN.export));
  else { if (!picker) caseShell(); drawCase(); }
}

async function main() {
  [cohort, metrics, index] = await Promise.all([get("cohort.json"), get("metrics.json"), get("cases/index.json")]);
  const status = document.getElementById("status");
  if (location.protocol === "file:")
    status.replaceChildren("Open this page through the local server so it can read the results: ",
      h("code", {}, "uv run python -m http.server -d ~/Desktop/hackathon 8000"), ", then http://localhost:8000/web/");
  else if (metrics)
    status.textContent = metrics.split === "test"
      ? "Results on the held-out test split, run once."
      : "Results on the validation split. The test split has not been touched.";
  else status.textContent = "No model has been evaluated yet. Sections without results say what to run.";
  renderAll();
  let raf;
  addEventListener("resize", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(renderAll); });
}
main();
