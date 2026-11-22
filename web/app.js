"use strict";
// Renders only what is in reports/web/*.json (contract: PLAN.md "Results frontend"). Nothing is computed from guesses.

const q = new URLSearchParams(location.search);
const DATA = "../reports/web/";
// ?theme=light|dark overrides the system setting (projectors, screenshots)
if (q.get("theme")) document.documentElement.dataset.theme = q.get("theme");

const RUN = {
  export: "uv run python -m orew.export",
  evaluate: "uv run python -m orew.evaluate --split val",
  illustrative: "uv run python -m orew.illustrative",
};
// ?illustrative=1 swaps in MADE-UP hypoxemia numbers (web/illustrative_hypoxemia.json) to show the layout with more
// events. Every hypoxemia block, chart, table and tooltip then carries ILL; hypotension always stays real.
const ILL = "ILLUSTRATIVE — NOT A REAL RESULT";
const EVENTS = ["hypotension", "hypoxemia"];
const BLOCKS = ["n", "overall", "overall_excl_flagged", "bins", "bins_excl_flagged", "subgroups", "reliability", "ablation"];
let ill = null, realN, realSplit; // the file, once loaded and checked; the real hypoxemia counts it replaced
const fake = (ev) => !!ill && ev === "hypoxemia";
const SIGNALS = {
  map: "Arterial MAP", sbp: "Arterial systolic", dbp: "Arterial diastolic", nibp_map: "Cuff MAP (NIBP)",
  hr: "Heart rate", spo2: "SpO2", etco2: "End-tidal CO2", rr: "Respiratory rate", fio2: "FiO2",
  tv: "Tidal volume", peep: "PEEP", pip: "Peak airway pressure", sevo: "Sevoflurane", des: "Desflurane",
  mac: "MAC", ppf_rate: "Propofol infusion", rftn_rate: "Remifentanil infusion", rftn50_rate: "Remifentanil infusion, 50 µg/mL", rftn_ug_h: "Remifentanil infusion",
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
const fill = (id, ...nodes) => document.querySelector(`#${id} .body`).replaceChildren(...nodes.flat(9).filter((n) => n != null));
// each word of a command is unbreakable so "-m" never splits from "python" on a phone
const empty = (what, cmd) => h("p", { class: "empty" }, `${what} Run `,
  h("code", {}, cmd.split(" ").map((w, i) => [i ? " " : null, h("span", { class: "nw" }, w)])), " to fill it.");
const fmt = (v, d = 0) => (v == null ? "n/a" : v.toLocaleString("en", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v, d = 0) => (v == null ? "n/a" : `${fmt(v * 100, d)}\u00a0%`); // no-break space: "%" never wraps alone
// a probability in %, to 2 significant digits: hypoxemia risks sit well below 1 %
const prob = (v, unit = "") => (v == null ? "n/a" : (v * 100).toLocaleString("en", { maximumSignificantDigits: 2 }) + unit);
const counts = (o, name = (k) => k, sortByKey) =>
  Object.entries(o).sort(sortByKey ? (a, b) => a[0].localeCompare(b[0]) : (a, b) => b[1] - a[1])
    .map(([k, n]) => `${name(k)}: ${n}`).join(", ");

async function get(path, base = DATA) {
  try {
    const r = await fetch(base + path);
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
    facts.push([`${cap(ev)} events`, `${fmt(e.n_events)} in ${fmt(e.n_cases)} of ${fmt(c.n_labelled)} labelled surgeries`]);
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
      h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Signal"), h("td", { class: "bar" }),
        h("th", { scope: "col" }, "Of time"), h("th", { scope: "col" }, "Of surgeries"))),
      h("tbody", {}, rows))));
}
const cap = (t) => t[0].toUpperCase() + t.slice(1);

// ---- chart helpers
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
function niceStep(lo, hi, n) {
  const raw = (hi - lo) / n, p = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((st) => st >= raw);
}
function ticks(lo, hi, n = 4, step = niceStep(lo, hi, n)) {
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
  { key: "map", label: "Arterial MAP (line) and cuff MAP (circles), mmHg", short: "MAP, mmHg (circles: cuff)", unit: "mmHg", clamp: [20, 200], thr: 65, h: 170, event: "hypotension" },
  { key: "spo2", label: "SpO2, %", unit: "%", clamp: [50, 100], thr: 90, h: 90, event: "hypoxemia" },
  { key: "hr", label: "Heart rate, bpm", unit: "bpm", clamp: [20, 220], h: 80 },
  { key: "etco2", label: "End-tidal CO2, mmHg", unit: "mmHg", clamp: [0, 100], h: 70 },
  { key: "bis", label: "BIS", unit: "", clamp: [0, 100], h: 70 },
  { key: "ppf_rate", label: "Propofol infusion, mL/h", unit: "mL/h", clamp: [0, 1000], h: 60, cls: "drug", zero: true },
  { key: "rftn_ug_h", label: "Remifentanil infusion, µg/h", unit: "µg/h", clamp: [0, 50000], h: 60, cls: "drug", zero: true },
];
// what each event is on the monitor, for the explorer's sentences
const EV = {
  hypotension: { sig: "MAP", unit: "mmHg", thr: 65, dur: "60 s", one: "drop", many: "drops" },
  hypoxemia: { sig: "SpO2", unit: "%", thr: 90, dur: "2 min", one: "desaturation", many: "desaturations" },
};
const EXPLORATORY = "exploratory (few events)";
const SPLIT = { train: "training", val: "validation", test: "test, unseen" };
const PHASES = [["Induction", "anestart", "opstart"], ["Surgery", "opstart", "opend"], ["Emergence", "opend", "aneend"]];
const RISK_H = { hypotension: 90, hypoxemia: 60 }, LANE_H = 48;
let caseId, caseData, picker, zoom = null, featured = [];

// minutes from case start -> h:mm (h:mm:ss with sec)
function hm(t, sec) {
  const a = Math.round(Math.abs(t) * (sec ? 60 : 1)), m = sec ? Math.floor(a / 60) : a;
  return `${t < 0 ? "−" : ""}${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}${sec ? `:${String(a % 60).padStart(2, "0")}` : ""}`;
}
const dur = (min) => { const m = Math.round(min); return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")}`; };
const mins = (v) => `${fmt(v, 1)} min`;
const plural = (n, E) => (n === 1 ? E.one : E.many);
const tag = (t) => h("span", { class: "tag" }, t);
// #case=<id> opens a surgery; the picker writes it back, so a link opens the same one
const hashCase = () => +(/^#case=(\d+)$/.exec(location.hash)?.[1] || 0) || null;

function caseShell() {
  const op = (c) => c.opname?.trim() || "operation not recorded";
  picker = h("select", { id: "case-pick" }, index.map((c) =>
    h("option", { value: c.caseid }, `${c.caseid} · ${SPLIT[c.split] || "split unknown"} · ${op(c)} · ${dur(c.duration_min)}`)));
  picker.onchange = () => openCase(+picker.value);
  const nTest = index.filter((c) => c.split === "test").length;
  const feat = featured.map((f) => ({ ...f, c: index.find((c) => c.caseid === f.caseid) })).filter((f) => f.c);
  caseId = index.some((c) => c.caseid === hashCase()) ? hashCase() : index[0].caseid;
  picker.value = caseId;
  fill("cases",
    h("p", {}, `${index.length} surgeries exported${nTest ? `, ${nTest === index.length ? "all" : nTest} from the test split, which the model never saw` : ""}. Each is shown whole, time in hours:minutes from case start; the signals are the raw monitor numbers before artifact cleaning, in 10 s steps.`),
    feat.length ? h("div", { class: "featured" },
      h("h3", {}, "Unseen test surgeries"),
      h("p", { class: "note" }, "Picked to show the model at its best and at its worst. Every other exported surgery is in the list below."),
      h("ul", {}, feat.map((f) => h("li", {}, h("a", { href: `#case=${f.caseid}`, "data-case": f.caseid },
        h("strong", {}, `${op(f.c)}`), h("span", { class: "feat-id" }, ` · surgery ${f.caseid}`), h("span", { class: "feat-why" }, f.reason)))))) : null,
    h("div", { class: "pick" }, h("label", { for: "case-pick" }, "Surgery"), picker),
    h("div", { id: "case-view" }));
}

function openCase(id, scroll) {
  if (!index.some((c) => c.caseid === id)) return;
  caseId = id; zoom = null; picker.value = id;
  if (location.hash !== `#case=${id}`) history.replaceState(null, "", `#case=${id}`);
  drawCase();
  if (scroll) document.getElementById("cases").scrollIntoView();
}
addEventListener("hashchange", () => { if (picker && hashCase()) openCase(hashCase(), true); });

function setZoom(z) {
  zoom = z;
  drawCase().then(() => document.getElementById(z ? "unzoom" : "case-chart")?.focus());
}

async function drawCase() {
  const id = caseId, view = document.getElementById("case-view");
  if (caseData?.caseid !== id) {
    const c = await get(`cases/${id}.json`);
    if (id !== caseId) return; // another surgery was picked while this one loaded
    caseData = c;
  }
  const c = caseData;
  for (const a of document.querySelectorAll(".featured a")) a.toggleAttribute("aria-current", +a.dataset.case === id);
  if (!c) return view.replaceChildren(empty(`Surgery ${id} is listed but its file is missing.`, RUN.export));
  const d = c.demographics;
  const who = [
    { F: "Female", M: "Male" }[d.sex] || "Sex not recorded", d.age != null && `${fmt(d.age)} years`,
    d.bmi != null && `BMI ${fmt(d.bmi, 1)}`, d.asa != null && `ASA ${fmt(d.asa)}`, d.emop ? "emergency" : "planned",
  ].filter(Boolean).join(", ");
  const notes = [];
  if (!c.events) notes.push(empty("Events are not labelled yet.", "uv run python -m orew.labels && " + RUN.export));
  if (!c.risk) notes.push(h("p", { class: "empty" }, "No risk curves yet. They appear once the model is trained and explain.py's records are exported."));
  else notes.push(h("p", { class: "note" }, {
    test: "Test surgery: this patient was never used to train or calibrate the model or to set its alert threshold.",
    val: "Validation surgery: the calibration and the alert threshold were fit on this split.",
    train: "Training surgery: its risk curves come from cross-fitted models that never saw this patient. Alarm reasons were computed for validation and test surgeries only.",
  }[c.split] || ""));
  const readout = h("p", { class: "readout", "aria-live": "polite" }, "Point at the chart, or focus it and use the arrow keys, to read values.");
  view.replaceChildren(...[
    h("p", { class: "who" }, h("strong", {}, `Surgery ${c.caseid}`), ` · ${who}. ${d.department || ""}${d.opname ? `: ${d.opname.trim()}` : ""}${d.optype ? ` (${d.optype})` : ""}. ${dur(c.duration_min)}.`,
      c.split ? ` ${cap(SPLIT[c.split] || c.split)} split.` : null),
    verdicts(c), ...notes,
    zoom ? h("p", { class: "zoombar" }, `Zoomed in to ${hm(zoom[0])}–${hm(zoom[1])}. `,
      h("button", { type: "button", id: "unzoom" }, "Show the whole surgery")) : null,
    readout, caseChart(c, view.clientWidth, readout), eventList(c), alertList(c)].flat(9).filter((n) => n != null));
  document.getElementById("unzoom")?.addEventListener("click", () => setZoom(null));
}

// the one-line answer per event: how many were warned, how early, and at what cost in false alarms
function verdicts(c) {
  if (!c.detection) return null;
  return h("div", { class: "verdicts" }, EVENTS.filter((ev) => c.detection[ev]).map((ev) => {
    const d = c.detection[ev], E = EV[ev], minor = ev === "hypoxemia";
    if (minor && !d.events && !d.alarms) return null;
    return h("p", { class: minor ? "verdict-line minor" : "verdict-line" }, h("strong", {}, cap(ev)), minor ? [" ", tag(EXPLORATORY)] : null, ": ",
      d.events ? [h("strong", {}, `${d.warned} of ${d.events} ${plural(d.events, E)} warned`), d.warned ? `, median ${mins(d.median_lead_min)} ahead` : ""] : `no ${E.many} in this surgery`,
      ` · ${d.false_alarms} false ${d.false_alarms === 1 ? "alarm" : "alarms"} of ${d.alarms} in ${fmt(d.hours, 1)} h monitored.`,
      d.not_warnable ? ` ${d.not_warnable} more ${plural(d.not_warnable, E)} started before the first prediction (first 10 min), not counted.` : null);
  }));
}

// text labels placed left to right without overlap: the first candidate centre (given the label width) that fits wins
function placer(lo, hi) {
  let free = -Infinity;
  return (text, centres, attrs) => {
    const w = text.length * 7.4 + 6;
    for (const c of centres(w)) {
      const l = Math.max(lo, Math.min(c - w / 2, hi - w));
      if (l >= free + 6) { free = l + w; return s("text", { ...attrs, x: l + w / 2, "text-anchor": "middle" }, text); }
    }
    return null;
  };
}

function caseChart(c, W, readout) {
  const ML = 56, MR = 14, TOP = 46, GAP = 32, narrow = W < 600; // narrow: titles start at the left edge, shorter, and sit higher to clear the top tick label
  // anestart can precede casestart and aneend follow caseend, so the axis spans every marker
  const marks = Object.values(c.phases_min).filter((v) => v != null);
  const T = c.t_min, end = Math.max(c.duration_min || 0, T[T.length - 1] || 1, ...marks), start = Math.min(0, ...marks);
  const [t0, tMax] = zoom ? [Math.max(start, zoom[0]), Math.min(end, zoom[1])] : [start, end];
  const x = lin(t0, tMax, ML, W - MR);
  const panels = PANELS.filter((p) => finite(c.series[p.key]).length);
  const risks = Object.entries(c.risk || {});
  const sig = (k) => { const p = panels.find((p) => p.key === k); return p && { ...p, vals: c.series[k] }; };
  const risk = (ev) => c.risk?.[ev] && { key: `risk-${ev}`, label: `${cap(ev)} risk in the next 10 min, %`, sub: ev === "hypoxemia" ? (narrow ? "exploratory" : EXPLORATORY) : null,
    unit: "%", clamp: [0, 1], h: RISK_H[ev], cls: "risk", zero: true, risk: c.risk[ev], ev };
  const lane = (ev) => (c.detection?.[ev]?.events || c.detection?.[ev]?.alarms) && { key: `lane-${ev}`, lane: ev, h: LANE_H,
    label: narrow ? `${cap(ev)}: alarms vs actual start` : `${cap(ev)}: model alarms and when each ${EV[ev].one} actually started`,
    sub: ev === "hypoxemia" ? (narrow ? "exploratory" : EXPLORATORY) : null };
  // hypotension first: its alarms, MAP and risk sit together at the top
  let yTop = TOP;
  const rows = [lane("hypotension"), sig("map"), risk("hypotension"), lane("hypoxemia"), sig("spo2"), risk("hypoxemia"),
    ...["hr", "etco2", "bis", "ppf_rate", "rftn_ug_h"].map(sig)].filter(Boolean)
    .map((p) => { const r = { ...p, y0: yTop + (narrow ? 26 : 20) }; yTop = r.y0 + p.h + GAP; return r; });
  const H = yTop + 4;
  const tStep = [1, 2, 5, 10, 15, 30, 60, 120].find((st) => x(st) - x(0) >= 52) || 120;
  const minor = { 1: 0.5, 2: 1, 5: 1, 10: 5, 15: 5, 30: 5, 60: 5, 120: 10 }[tStep];
  const g = [], plot = []; // plot: everything placed in time, clipped to the plot area when zoomed in
  const onGrid = (t, st) => Math.abs(t / st - Math.round(t / st)) < 1e-6;

  // anesthesia-record grid: faint minor columns, stronger every tStep, across every panel
  for (const r of rows)
    for (let t = Math.ceil(t0 / minor) * minor; t <= tMax + 1e-9; t += minor)
      plot.push(s("line", { class: onGrid(t, tStep) ? "grid-major" : "grid", x1: x(t), x2: x(t), y1: r.y0, y2: r.y0 + r.h }));
  for (const y of [TOP - 6, H - 6]) {
    g.push(s("text", { x: 0, y, class: "axis-unit" }, "h:mm"));
    for (let t = Math.ceil(t0 / tStep) * tStep; t <= tMax + 1e-9; t += tStep) g.push(s("text", { x: x(t), y, "text-anchor": "middle" }, hm(t)));
  }

  // phase strip + marker lines
  const ph = c.phases_min;
  for (const [name, a, b] of PHASES) {
    if (ph[a] == null || ph[b] == null) continue;
    const l = Math.max(x(ph[a]), ML), w = Math.min(x(ph[b]), W - MR) - l;
    if (w <= 0) continue;
    plot.push(s("rect", { class: `phase-${a}`, x: l, y: 4, width: Math.max(w, 1), height: 18, rx: 2 },
      s("title", {}, `${name}: ${hm(ph[a])}–${hm(ph[b])}`)));
    if (w > name.length * 8 + 8) plot.push(s("text", { class: "phase-label", x: l + w / 2, y: 17, "text-anchor": "middle" }, name));
  }
  // marker lines per panel, so they don't cross the panel titles
  for (const r of rows) for (const k of ["anestart", "opstart", "opend", "aneend"])
    if (ph[k] != null) plot.push(s("line", { class: "phase", x1: x(ph[k]), x2: x(ph[k]), y1: r.y0, y2: r.y0 + r.h }));

  const alertsOf = (ev) => (c.alerts || []).filter((a) => a.type === ev);
  const evOf = (ev) => (c.events || []).filter((e) => e.type === ev);
  const alarmAt = (e) => e.alarm_min != null && (c.alerts || []).find((a) => a.type === e.type && a.t_min === e.alarm_min);
  const summary = [];
  for (const r of rows) {
    g.push(s("text", { class: "title", x: narrow ? 0 : ML, y: r.y0 - (narrow ? 12 : 7) }, (narrow && r.short) || r.label, r.sub ? s("tspan", { class: "row-sub" }, `  ${r.sub}`) : null));
    const clip = `clip-${r.key}`, inner = [];
    g.push(s("clipPath", { id: clip }, s("rect", { x: ML, y: r.y0 - 2, width: W - MR - ML, height: r.h + 4 })));
    if (r.lane) { // alarms on top, a bracket from the warning alarm to the onset, the onset as a line
      const E = EV[r.lane], yb = r.y0 + 32, labels = [], place = placer(ML, W - MR);
      // a label whose bracket centre is outside a zoomed window would sit at the edge, next to the wrong marks
      const label = (cx, text, cls) => cx >= ML && cx <= W - MR && labels.push(place(text, () => [cx], { class: cls, y: yb - 7 }));
      let bEnd = -Infinity; // a bracket starting inside the previous one (one alarm warned both) drops to a second row
      inner.push(s("line", { class: "grid-major", x1: ML, x2: W - MR, y1: r.y0 + r.h, y2: r.y0 + r.h }));
      for (const e of evOf(r.lane)) {
        const xo = x(e.onset_min), a = alarmAt(e);
        inner.push(s("line", { class: `onset onset-${e.type}`, x1: xo, x2: xo, y1: r.y0 + 14, y2: r.y0 + r.h },
          s("title", {}, `${cap(E.one)} started at ${hm(e.onset_min, true)}`)));
        if (a) {
          const xa = x(a.t_min), row2 = xa < bEnd - 1, y = row2 ? yb + 10 : yb;
          bEnd = Math.max(bEnd, xo);
          inner.push(s("path", { class: "bracket", d: `M${xa},${y - 4}v8M${xa},${y}H${xo - 1}m-5,-3.5l5,3.5l-5,3.5` },
            s("title", {}, `Warned ${mins(e.lead_min)} before the ${E.one}`)));
          if (!row2) label((xa + xo) / 2, zoom ? `warned ${fmt(e.lead_min, 1)} min earlier` : `${fmt(e.lead_min, 1)} min`, "lead-label");
        } else if (e.warnable) {
          inner.push(s("path", { class: "missed", d: `M${xo - 5},${yb - 5}l10,10m0,-10l-10,10` }, s("title", {}, `Missed: no alarm in the 10 min before`)));
          label(xo, "missed", "missed-label");
        }
      }
      for (const a of alertsOf(r.lane))
        inner.push(s("path", { class: a.true === false ? "alert false" : "alert", d: `M${x(a.t_min)},${r.y0 + 13}l-6,-11h12z` },
          s("title", {}, `Alarm at ${hm(a.t_min, true)}: ${a.true === false ? `false alarm, no ${E.one} within 10 min` : `a ${E.one} followed within 10 min`}`)));
      inner.push(labels);
      g.push(s("g", { "clip-path": `url(#${clip})` }, inner));
      continue;
    }
    const vals = r.risk ? r.risk.p10 : r.vals, ts = r.risk ? r.risk.t_min : T;
    const seen = finite(vals).filter((v) => v >= r.clamp[0] && v <= r.clamp[1]);
    let lo = r.zero ? 0 : Math.min(...seen, r.thr ?? Infinity), hi = Math.max(...seen, r.thr ?? -Infinity, r.zero ? 1 : -Infinity);
    // brief bolus rates (often 10x the infusion) would flatten the infusion itself to the floor
    const sorted = r.cls === "drug" && [...seen].sort((a, b) => a - b), cut = sorted && Math.max(1, sorted[Math.floor(sorted.length * 0.98)]);
    const clipped = cut && hi > cut;
    if (clipped) hi = cut;
    if (r.risk) [lo, hi] = [0, Math.min(1, 1.2 * Math.max(0.001, r.risk.threshold ?? 0, ...seen))];
    const n = r.h > 90 ? 4 : r.h >= 80 ? 3 : 2, st = niceStep(lo, hi || 1, n);
    [lo, hi] = [Math.floor(lo / st) * st, Math.ceil(hi / st) * st || 1];
    const tk = ticks(lo, hi, n, st), y = lin(lo, hi, r.y0 + r.h, r.y0);
    if (clipped && !narrow) g.push(s("text", { class: "row-sub", x: W - MR, y: r.y0 - 7, "text-anchor": "end" }, "brief peaks run off the top"));
    for (const v of tk) {
      g.push(s("line", { class: "grid", x1: ML, x2: W - MR, y1: y(v), y2: y(v) }));
      g.push(s("text", { x: ML - 6, y: y(v) + 4, "text-anchor": "end" }, r.risk ? prob(v) : `${v}`));
    }
    for (const e of evOf(r.event || r.ev))
      inner.push(s("rect", { class: `ev ev-${e.type}`, x: x(e.onset_min), y: r.y0, width: Math.max(x(e.end_min) - x(e.onset_min), 2), height: r.h },
        s("title", {}, `${cap(e.type)} ${hm(e.onset_min, true)}–${hm(e.end_min, true)}`)));
    const thr = r.thr ?? r.risk?.threshold;
    if (thr != null) inner.push(s("line", { class: "thr", x1: ML, x2: W - MR, y1: y(thr), y2: y(thr) }));
    inner.push(s("path", { class: `line s-${r.cls || r.key}`, d: pathD(ts, vals, x, y) }));
    if (r.key === "map") // a cuff reading is held for up to 30 s on the grid, so draw only where the value changes
      c.series.nibp_map.forEach((v, i, a) => v != null && v !== a[i - 1] && inner.push(s("circle", { class: "nibp", cx: x(T[i]), cy: y(v), r: 3 })));
    // the value the warning alarm saw: what the current number looked like while the model already warned
    const place = placer(ML, W - MR), done = new Set(); // one alarm can warn two events: one dot
    if (r.event) for (const e of evOf(r.event)) {
      const a = alarmAt(e);
      if (!a || a.now == null || done.has(a)) continue;
      done.add(a);
      const xa = x(a.t_min), ya = y(Math.max(lo, Math.min(hi, a.now)));
      inner.push(s("line", { class: "alarm-line", x1: xa, x2: xa, y1: r.y0, y2: ya }),
        s("circle", { class: "alarm-dot", cx: xa, cy: ya, r: 4.5 }, s("title", {}, `${EV[r.event].sig} ${fmt(a.now)} ${r.unit} when the model warned`)));
      if (zoom && a.t_min >= t0 && a.t_min <= tMax) inner.push(place(`${EV[r.event].sig} ${fmt(a.now)} at the alarm`, (w) => [xa - 8 - w / 2, xa + 8 + w / 2], { class: "thr-label", y: ya - 9 }));
    }
    for (const a of (r.risk ? alertsOf(r.ev) : []))
      inner.push(s("path", { class: a.true === false ? "alert false" : "alert", d: `M${x(a.t_min)},${r.y0 + 10}l-5,-9h10z` }, s("title", {}, `Alarm at ${hm(a.t_min, true)}`)));
    if (thr != null) inner.push(s("text", { class: "thr-label", x: W - MR - 4, y: y(thr) - 5, "text-anchor": "end" }, r.risk ? `alarm at ${prob(thr, " %")}` : `${thr} ${r.unit}`));
    g.push(s("g", { "clip-path": `url(#${clip})` }, inner));
    const f = r.risk ? prob : fmt;
    summary.push(`${r.label}: ${seen.length ? `median ${f(median(seen))}, range ${f(Math.min(...seen))} to ${f(Math.max(...seen))}` : "no data"}.`);
  }
  g.unshift(s("clipPath", { id: "clip-plot" }, s("rect", { x: ML, y: 0, width: W - MR - ML, height: H })), s("g", { "clip-path": "url(#clip-plot)" }, plot));

  // crosshair readout, by pointer or arrow keys
  const cursor = s("line", { class: "cursor", y1: 22, y2: H - 20, visibility: "hidden" });
  const svg = s("svg", { id: "case-chart", width: W, height: H, viewBox: `0 0 ${W} ${H}`, tabindex: 0, role: "img",
    "aria-label": `Vital signs ${zoom ? `from ${hm(t0)} to ${hm(tMax)}` : `over ${dur(tMax - t0)}`} of surgery ${c.caseid}. ${summary.join(" ")} The list below the chart gives each event and the model's first alarm before it. Use the left and right arrow keys to read values.` },
  g, cursor);
  let at = -1;
  const show = (i) => {
    at = Math.max(0, Math.min(T.length - 1, i));
    const t = T[at], parts = [hm(t, true)];
    for (const p of panels) {
      const v = c.series[p.key][at];
      parts.push(`${SIGNALS[p.key] || p.label} ${v == null ? "no value" : `${fmt(v)} ${p.unit}`.trim()}`);
      if (p.key === "map" && c.series.nibp_map[at] != null) parts.push(`cuff MAP ${fmt(c.series.nibp_map[at])} mmHg`);
    }
    for (const [ev, r] of risks) {
      let j = r.t_min.findLastIndex((u) => u <= t);
      if (j >= 0 && r.p10[j] != null) parts.push(`${ev} 10 min risk ${prob(r.p10[j], " %")}`);
    }
    readout.textContent = parts.join(", ");
    cursor.setAttribute("x1", x(t)); cursor.setAttribute("x2", x(t)); cursor.setAttribute("visibility", "visible");
  };
  const nearest = (t) => { let i = T.findIndex((u) => u >= t); return i < 0 ? T.length - 1 : i; };
  const toT = (px) => t0 + ((px - ML) / (W - MR - ML)) * (tMax - t0);
  // on the svg itself, not a hit rect over it, so the marks' <title> tooltips stay reachable; pointerdown for taps
  for (const type of ["pointermove", "pointerdown"])
    svg.addEventListener(type, (e) => show(nearest(toT(e.clientX - svg.getBoundingClientRect().left))));
  svg.addEventListener("keydown", (e) => {
    const step = { ArrowRight: 6, ArrowLeft: -6, PageDown: 60, PageUp: -60 }[e.key]; // 6 bins = 1 min
    if (step) { e.preventDefault(); show((at < 0 ? nearest(t0) : at) + (e.shiftKey ? step * 10 : step)); }
    if (e.key === "Home" || e.key === "End") { e.preventDefault(); show(nearest(e.key === "Home" ? t0 : tMax)); }
  });
  const missing = PANELS.filter((p) => !panels.includes(p)).map((p) => SIGNALS[p.key]);
  return h("div", { class: "chart" }, svg, h("ul", { class: "vh" }, summary.map((t) => h("li", {}, t))),
    c.detection ? h("p", { class: "legend chart-key" },
      h("span", { class: "k-alarm", "aria-hidden": "true" }, "▼"), " model alarm, an event followed within 10 min · ",
      h("span", { class: "k-false", "aria-hidden": "true" }, "▽"), " false alarm · ",
      h("span", { class: "k-onset", "aria-hidden": "true" }, "│"), " the event actually starts · ",
      h("span", { class: "k-bracket", "aria-hidden": "true" }, "├─▶"), " warning time · ",
      h("span", { class: "k-dot", "aria-hidden": "true" }, "●"), " MAP or SpO2 when the model warned. Shaded: the event; dashed lines: thresholds.") : null,
    missing.length ? h("p", { class: "note" }, `Not recorded in this surgery: ${missing.join(", ")}.`) : null);
}

// each event in words, with a zoom button: the chart's labels get crowded over a whole surgery
function eventList(c) {
  if (!c.events?.length) return c.events ? h("p", { class: "note" }, "No labelled events in this surgery.") : null;
  return EVENTS.map((ev) => {
    const evs = c.events.filter((e) => e.type === ev), E = EV[ev];
    if (!evs.length) return null;
    return h("div", { class: "events" },
      h("h3", {}, `${cap(ev)}: when each ${E.one} started, and the model's first alarm before it`, ev === "hypoxemia" ? [" ", tag(EXPLORATORY)] : null),
      h("p", { class: "note" }, `A ${E.one} is ${E.sig} under ${E.thr} ${E.unit} for ${E.dur}. It counts as warned if an alarm fired in the 10 minutes before it started (the evaluation's rule), so one alarm can count for back-to-back ${E.many}.`),
      h("ol", {}, evs.map((e, i) => {
        const a = e.alarm_min != null && c.alerts?.find((x) => x.type === ev && x.t_min === e.alarm_min);
        // above the threshold just after a drop ended is a recovery, not a normal-looking number
        const p = a ? evs.findLastIndex((q) => q.end_min <= a.t_min) : -1, gap = p >= 0 ? a.t_min - evs[p].end_min : Infinity;
        const between = a ? evs.flatMap((q, j) => (q.onset_min > a.t_min && q.onset_min < e.onset_min ? [j + 1] : [])) : [];
        const what = a ? [h("strong", {}, `Warned ${mins(e.lead_min)} earlier`), ` (alarm at ${hm(a.t_min, true)}).`,
          a.now != null ? ` ${E.sig} at the alarm: ${fmt(a.now)} ${E.unit}, ${a.now > E.thr ? (gap <= 5 ? `back above ${E.thr} only ${gap < 1 ? `${Math.round(gap * 60)} s` : mins(gap)} after ${E.one} ${p + 1} ended.` : `still above ${E.thr}: the number looked normal.`) : a.now === E.thr ? `right at ${E.thr}, not under it yet.` : `already under ${E.thr}, but not yet for ${E.dur}.`}` : null,
          between.length ? ` This alarm also came before ${plural(between.length, E)} ${between.join(" and ")}.` : null]
          : e.warnable === false ? [h("strong", {}, "Not scored"), ": it started before the model's first prediction (first 10 min)."]
          : e.warnable ? [h("strong", { class: "miss" }, "Missed"), ": no alarm in the 10 minutes before it."]
          : "No risk curves for this surgery.";
        const zoomTo = [e.onset_min - 15, e.onset_min + 10];
        return h("li", {}, h("span", { class: "when" }, `${cap(E.one)} ${i + 1} at ${hm(e.onset_min, true)}`),
          h("span", { class: "what" }, what, ` It lasted ${mins(e.end_min - e.onset_min)}.`),
          h("button", { type: "button", class: "zoom", "data-zoom": zoomTo.join(","), "aria-label": `Zoom the chart to ${E.one} ${i + 1}` }, "Zoom in"));
      })));
  });
}
addEventListener("click", (e) => { const b = e.target.closest?.("button[data-zoom]"); if (b) setZoom(b.dataset.zoom.split(",").map(Number)); });

function alertList(c) {
  if (!c.alerts?.length) return null;
  return EVENTS.map((ev) => {
    const al = c.alerts.filter((a) => a.type === ev), E = EV[ev];
    if (!al.length) return null;
    return h("details", { class: "alarms" },
      h("summary", {}, `All ${al.length} ${ev} alarms, with the model's reasons`, ev === "hypoxemia" ? ` — ${EXPLORATORY}` : ""),
      h("ol", {}, al.map((a) => h("li", {}, h("strong", {}, hm(a.t_min, true)), ` · risk ${prob(a.p10, " %")}`,
        a.true == null ? null : a.true ? ` · true alarm: a ${E.one} followed within 10 min` : h("span", { class: "miss" }, ` · false alarm: no ${E.one} within 10 min`),
        a.now != null ? ` · ${E.sig} ${fmt(a.now)} ${E.unit}` : null, ".",
        a.lead_specialist ? ` Led by the ${a.lead_specialist} model.` : null,
        a.top_drivers?.length ? ` Reasons: ${a.top_drivers.join("; ")}.` : null))));
  });
}

// ---- results from metrics.json
const MODEL = {
  baseline1: "Baseline 1: current value", baseline2: "Baseline 2: value and 5 min slope",
  snapshot: "Snapshot (XGBoost)", trajectory: "Trajectory (GRU and CNN)", pattern: "Pattern", fusion: "Fusion",
};
const modelName = (m) => (MODEL[m] || m) + (m === metrics.final_model ? " (final)" : "");
const SIGNAL_OF = { hypotension: "MAP", hypoxemia: "SpO2" };
const binLabel = (ev, l) => `${SIGNAL_OF[ev]} ${l.replace(">=", "≥ ").replace(">", "> ").replace("-", "–")}`;
const ciText = (m, d = 2, f = (v) => fmt(v, d)) =>
  m == null ? "n/a" : [f(m[0]), h("span", { class: "ci" }, `${f(m[1])}–${f(m[2])}`)];
const HH = () => String(metrics.headline_horizon_min);

// Rows of [label, sub, final, baseline] intervals on one shared x scale: filled circle = final model, open = baseline 1.
function dumbbell(title, rows, W, ciPair, isFake) {
  const ML = 8, MR = 16, RH = 46, off = isFake ? 18 : 0, TOP = 22 + off;
  const all = rows.flatMap((r) => [r.a, r.b]).filter(Boolean).flatMap((m) => m.slice(0, 3));
  const lo = Math.floor(Math.min(...all) * 10) / 10, hi = Math.min(1, Math.ceil(Math.max(...all) * 10) / 10);
  const x = lin(lo, hi, ML, W - MR), H = TOP + rows.length * RH + 22;
  const g = [isFake ? illMark(W, H, ML, 12) : null, s("text", { class: "title", x: ML, y: 12 + off }, title)];
  for (const v of ticks(lo, hi, W < 400 ? 3 : 5)) {
    g.push(s("line", { class: "grid", x1: x(v), x2: x(v), y1: TOP, y2: H - 20 }));
    g.push(s("text", { x: x(v), y: H - 4, "text-anchor": "middle" }, fmt(v, 2)));
  }
  rows.forEach((r, i) => {
    const y = TOP + i * RH;
    g.push(s("text", { x: ML, y: y + 14, class: "row-label" }, r.label, r.sub ? s("tspan", { class: "row-sub" }, `  ${r.sub}`) : null));
    for (const [m, dy, cls, who] of [[r.b, 24, "mk-base", ciPair[1]], [r.a, 36, "mk-final", ciPair[0]]]) {
      if (!m) continue;
      g.push(s("g", { class: cls },
        s("title", {}, `${who}: ${fmt(m[0], 3)} (95 % CI ${fmt(m[1], 3)}–${fmt(m[2], 3)})${isFake ? `. ${ILL}` : ""}`),
        s("line", { x1: x(m[1]), x2: x(m[2]), y1: y + dy, y2: y + dy }),
        s("circle", { cx: x(m[0]), cy: y + dy, r: 4.5 })));
    }
  });
  const table = h("div", { class: "vh" }, h("table", {}, h("caption", {}, isFake ? `${title}: ${ILL}` : title),
    h("tr", {}, h("th", {}, isFake ? `Group (${ILL})` : "Group"), h("th", {}, ciPair[0]), h("th", {}, ciPair[1])),
    rows.map((r) => h("tr", {}, h("th", {}, `${r.label} ${r.sub || ""}`), h("td", {}, ciText(r.a, 3)), h("td", {}, ciText(r.b, 3))))));
  return h("figure", { class: "dumbbell" }, s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, g), table);
}

const legend = () => h("p", { class: "legend" },
  h("span", { class: "key key-final", "aria-hidden": "true" }), `${modelName(metrics.final_model)}  `,
  h("span", { class: "key key-base", "aria-hidden": "true" }), `${MODEL.baseline1}. Bars are 95 % confidence intervals${metrics.bootstrap ? ` from ${fmt(metrics.bootstrap)} patient bootstrap resamples` : ""}.`, illCI());
// the made-up intervals come from a formula, so no line may call them bootstrap CIs
const illCI = () => (ill ? h("span", { class: "ill-tag" }, " Hypoxemia intervals are made up, not bootstrapped.") : null);

let illN = 0;
// stripes and the label drawn inside the svg itself, so even a cropped screenshot of one chart carries them
function illMark(W, H, x, y, lines = [ILL]) {
  const id = `ill-${++illN}`;
  return [s("pattern", { id, width: 12, height: 12, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" },
    s("rect", { class: "ill-stripe", width: 6, height: 12 })),
  s("rect", { width: W, height: H, fill: `url(#${id})` }),
  lines.map((t, i) => s("text", { class: "ill-label", x, y: y + i * 13 }, t))];
}
// every per-event block opens here, so neither the illustrative label nor the real small-sample caveat is skipped
function evHead(ev, title = cap(ev)) {
  if (fake(ev)) return [h("h3", {}, `${title} `, h("span", { class: "ill-tag" }, ILL)),
    h("p", { class: "ill-note" }, "Made-up numbers that show what hypoxemia results could look like with more surgeries. No model produced them.",
      realN ? ` The real ${realSplit === "test" ? "test" : "validation"} split has only ${fmt(realN.events)} hypoxemia events.` : null)];
  const n = metrics.n?.[ev]?.events;
  // 100: a few events per value band and subgroup at best, so intervals stay wide
  return [h("h3", {}, title), ev === "hypoxemia" && !(n >= 100)
    ? h("p", { class: "note" }, `Only ${n == null ? "a few" : fmt(n)} hypoxemia events in this split — too few for firm conclusions.`) : null];
}
const block = (ev, ...kids) => (fake(ev) ? h("div", { class: "ill" }, kids) : kids);
// with no metrics.json, illustrative mode still says hypotension has no results
const missing = (src) => (src ? EVENTS.filter((ev) => !src[ev]).map((ev) => [h("h3", {}, cap(ev)), empty("Not computed yet.", RUN.evaluate)]) : []);
const modelTh = (ev) => h("th", { scope: "col" }, "Model", fake(ev) ? h("span", { class: "ill-tag" }, ` (${ILL})`) : null);

let exclFlagged = false;
const flagged = (k) => metrics[exclFlagged ? `${k}_excl_flagged` : k];
// one setting for the headline and the metrics tables: both redraw, and focus returns to the box that was used
function flagBox(id) {
  const box = h("input", { type: "checkbox", id, checked: exclFlagged || null });
  box.onchange = () => { exclFlagged = box.checked; renderAll(); document.getElementById(id).focus(); };
  return h("p", { class: "check" }, box, h("label", { for: id }, "Leave out times within 5 min after an event ended"));
}

function headline(W) {
  const fm = metrics.final_model, H = HH(), byEv = flagged("bins"), out = [
    h("p", {}, `Discrimination ${H} minutes ahead, inside each band of the current value. Baseline 1 sees only the current number, which says little inside a narrow band. The model has to beat it in every band, including the bands where the number still looks normal.`),
    flagBox("excl-flagged-h")];
  if (!byEv) return fill("headline", out, empty("Value-band results not computed.", RUN.evaluate));
  out.push(legend(), ...missing(byEv));
  for (const ev of Object.keys(byEv)) {
    const bins = byEv[ev][H] || [], all = flagged("overall")?.[ev]?.[H] || {};
    const rows = (k) => [{ label: "All times", a: all[fm]?.[k], b: all.baseline1?.[k] },
      ...bins.map((b) => ({ label: binLabel(ev, b.label), sub: `${fmt(b.n_pos)} of ${fmt(b.n)} times before an onset`, a: b.models[fm]?.[k], b: b.models.baseline1?.[k] }))];
    const wins = bins.filter((b) => b.models[fm]?.auroc?.[0] > b.models.baseline1?.auroc?.[0]).length;
    const cw = W >= 760 ? (W - 32) / 2 : W;
    const o = all[fm], b1 = all.baseline1;
    out.push(block(ev, evHead(ev, `${cap(ev)}, by current ${SIGNAL_OF[ev]}`),
      h("p", { class: "verdict" }, fake(ev) ? `${ILL}: ` : null, `${modelName(fm)} has the higher AUROC in ${wins} of ${bins.length} ${SIGNAL_OF[ev]} bands (point estimates; check the intervals).`,
        o?.median_lead_min && b1?.median_lead_min
          ? ` Events warned in the 10 min before onset ${pct(o.event_recall?.[0])} against ${pct(b1.event_recall?.[0])}; median lead of those warned ${fmt(o.median_lead_min[0], 1)} min against ${fmt(b1.median_lead_min[0], 1)} min; events warned at least 5 min ahead ${pct(o.pct_warned_5min?.[0])} against ${pct(b1.pct_warned_5min?.[0])}.`
          : null),
      h("div", { class: "pair" },
        dumbbell("AUROC", rows("auroc"), cw, [modelName(fm), MODEL.baseline1], fake(ev)),
        dumbbell("AUPRC", rows("auprc"), cw, [modelName(fm), MODEL.baseline1], fake(ev)))));
  }
  fill("headline", ...out);
}

const DISC = [["auroc", "AUROC"], ["auprc", "AUPRC"], ["brier", "Brier score", 3], ["sens_at_90spec", "Sensitivity at 90 % specificity, %", 0, true]];
const cell = (m, d = 2, asPct) => h("td", {}, ciText(m, d, asPct ? (v) => fmt(v * 100) : undefined));

// heading and sizes sit outside the scroll box, so on a phone they wrap instead of scrolling away with the table
function evTable(ev, ...kids) {
  const n = metrics.n?.[ev];
  return block(ev, evHead(ev),
    h("p", { class: "note" }, fake(ev) ? h("span", { class: "ill-tag" }, `${ILL}: `) : null, n ? `${fmt(n.events)} events in ${fmt(n.cases)} surgeries (${fmt(n.subjects)} patients), ${fmt(n.times)} prediction times.` : null,
      h("span", { class: "scroll-hint" }, " Scroll the table sideways for more columns.")),
    h("div", { class: "scroll" }, h("table", { class: "metrics" },
      fake(ev) ? h("caption", { class: "ill-tag" }, `${cap(ev)}: ${ILL}`) : h("caption", { class: "vh" }, cap(ev)), kids)));
}

function metricTables() {
  const src = flagged("overall");
  if (!src) return [empty("Not computed.", RUN.evaluate)];
  return [...missing(src), Object.entries(src).map(([ev, byH]) => evTable(ev,
      h("thead", {}, h("tr", {}, modelTh(ev), DISC.map(([, l]) => h("th", { scope: "col" }, l)))),
      Object.entries(byH).map(([H, byM]) => h("tbody", {},
        h("tr", { class: "group" }, h("th", { scope: "rowgroup", colspan: DISC.length + 1 }, `${H} minutes ahead`)),
        metrics.models.filter((m) => byM[m]).map((m) => h("tr", { class: m === metrics.final_model ? "final" : null },
          h("th", { scope: "row" }, modelName(m)), DISC.map(([k, , d, p]) => cell(byM[m][k], d ?? 2, p))))))))];
}

function metricsSection() {
  const ab = metrics.ablation || {};
  fill("metrics",
    h("p", {}, `Every eligible prediction time, every 30 s. Intervals are 95 % ${metrics.bootstrap ? "patient-bootstrap " : ""}CIs.`, illCI(), " Brier score: lower is better."),
    flagBox("excl-flagged"),
    metricTables(),
    Object.keys(ab).length ? [
      h("h3", {}, "Drug-feature ablation"),
      h("p", { class: "note" }, "The same model trained with and without the infusion and anesthetic-agent features. Vasopressor infusions are rare in this cohort, so this mostly measures propofol and remifentanil."),
      h("div", { class: "scroll" }, h("table", {},
        ill ? h("caption", { class: "ill-tag" }, `Hypoxemia rows: ${ILL}`) : null,
        h("thead", {}, h("tr", {}, ["Event", "Ahead", "Model", "AUROC with drugs", "AUROC without", "AUPRC with drugs", "AUPRC without"].map((l) => h("th", { scope: "col" }, l)))),
        h("tbody", {}, Object.entries(ab).flatMap(([ev, byH]) => Object.entries(byH).map(([H, a]) =>
          h("tr", { class: fake(ev) ? "ill-row" : null }, h("th", { scope: "row" }, cap(ev), fake(ev) ? h("span", { class: "ill-tag" }, ` (${ILL})`) : null), h("td", {}, `${H} min`), h("td", {}, MODEL[a.model] || a.model),
            cell(a.with_drugs?.auroc), cell(a.without_drugs?.auroc), cell(a.with_drugs?.auprc), cell(a.without_drugs?.auprc)))))))]
      : h("p", { class: "note" }, "Drug-feature ablation not computed."));
}

function reliability(title, sub, pts, D, size, isFake) {
  const ML = 44, MB = 40, top = isFake ? 30 : 0, SH = size + top, x = lin(0, D, ML, size - 8), y = lin(0, D, SH - MB, 8 + top);
  const tk = [...new Set([...ticks(0, D, 2), D])], g = [isFake ? illMark(size, SH, 2, 12, ["ILLUSTRATIVE —", "NOT A REAL RESULT"]) : null]; // D too, so the top of each axis has a number
  for (const v of tk) {
    g.push(s("line", { class: "grid", x1: x(v), x2: x(v), y1: y(0), y2: y(D) }), s("line", { class: "grid", x1: x(0), x2: x(D), y1: y(v), y2: y(v) }));
    g.push(s("text", { x: x(v), y: y(0) + 16, "text-anchor": "middle" }, prob(v)), s("text", { x: x(0) - 5, y: y(v) + 4, "text-anchor": "end" }, prob(v)));
  }
  g.push(s("text", { x: (x(0) + x(D)) / 2, y: SH - 4, "text-anchor": "middle" }, "Predicted, %"),
    s("text", { transform: `translate(12 ${(y(0) + y(D)) / 2}) rotate(-90)`, "text-anchor": "middle" }, "Observed, %"));
  g.push(s("line", { class: "thr", x1: x(0), y1: y(0), x2: x(D), y2: y(D) }));
  g.push(s("path", { class: "line s-risk", d: pathD(pts.map((p) => p.p), pts.map((p) => Math.min(p.obs, D)), x, y) }));
  for (const p of pts) g.push(s("circle", { class: "rel-pt", cx: x(p.p), cy: y(Math.min(p.obs, D)), r: 3.5 },
    s("title", {}, `Predicted ${prob(p.p, "\u00a0%")}, observed ${prob(p.obs, "\u00a0%")}, ${fmt(p.n)} times${isFake ? `. ${ILL}` : ""}`)));
  return h("figure", { class: "rel", style: `width:${size}px` }, h("figcaption", {}, title, h("span", { class: "note" }, sub, isFake ? h("span", { class: "ill-tag" }, ` (${ILL})`) : null)),
    s("svg", { width: size, height: SH, viewBox: `0 0 ${size} ${SH}`, "aria-hidden": "true" }, g),
    h("div", { class: "vh" }, h("table", {}, h("caption", {}, isFake ? `${title}: ${ILL}` : title), h("tr", {}, h("th", {}, isFake ? `Predicted (${ILL})` : "Predicted"), h("th", {}, "Observed"), h("th", {}, "Times")),
      pts.map((p) => h("tr", {}, h("td", {}, prob(p.p, "\u00a0%")), h("td", {}, prob(p.obs, "\u00a0%")), h("td", {}, fmt(p.n)))))));
}

function calibration(W) {
  const H = HH(), out = [h("p", {}, `Observed against predicted risk of an onset in the next ${H} minutes, in bins of predicted risk, both in %. Points on the dashed diagonal are well calibrated; points below it mean the model overstates risk.`)];
  const size = Math.max(150, Math.min(230, Math.floor((W - 48) / 3)));
  out.push(...missing(metrics.reliability));
  for (const [ev, byH] of Object.entries(metrics.reliability || {})) {
    const byM = byH[H] || {}, all = Object.values(byM).flat();
    if (!all.length) continue;
    const D = Math.min(1, Math.ceil(Math.max(...all.flatMap((p) => [p.p, p.obs])) * 10) / 10);
    out.push(block(ev, evHead(ev), h("div", { class: "rels" }, metrics.models.filter((m) => byM[m]?.length).map((m) => {
      const b = metrics.overall?.[ev]?.[H]?.[m]?.brier;
      return reliability(modelName(m), b ? `Brier ${fmt(b[0], 3)}` : "Brier not computed", byM[m], D, size, fake(ev));
    }))));
  }
  fill("calibration", out.length > 1 ? out : empty("Reliability curves not computed.", RUN.evaluate));
}

const ALERT = [["alarms_per_hour", "Alarms per hour", 1], ["false_per_true_alarm", "False per true alarm", 1],
  ["median_lead_min", "Median lead, min", 1], ["pct_warned_5min", "Warned 5+ min ahead, %", 0, true],
  ["event_recall", "Event recall, %", 0, true]];

function alertsSection() {
  const H = HH(), a = metrics.alerts || {};
  const out = [h("p", {}, `An alert fires when a model's ${H}-minute risk crosses its threshold, then stays quiet for 5 minutes unless the risk rises further. `,
    a.chosen_on ? `Each threshold was chosen on the ${a.chosen_on === "val" ? "validation" : a.chosen_on} split so that every model alarms about ${fmt(a.alarms_per_hour_target, 1)} times per hour, which keeps the comparison fair. ` : null,
    "Event recall is the share of events with an alert in the 10 minutes before onset; repeat alarms within 5 minutes count once. Median lead runs from the first such alert and covers warned events only; the 5+ min share is of all events.", " An alarm is true if an onset follows within 10 minutes.")];
  out.push(...missing(metrics.overall));
  for (const [ev, byH] of Object.entries(metrics.overall || {})) {
    const byM = byH[H] || {};
    out.push(evTable(ev,
      h("thead", {}, h("tr", {}, modelTh(ev), h("th", { scope: "col" }, "Threshold, %"), ALERT.map(([, l]) => h("th", { scope: "col" }, l)))),
      h("tbody", {}, metrics.models.filter((m) => byM[m]).map((m) => h("tr", { class: m === metrics.final_model ? "final" : null },
        h("th", { scope: "row" }, modelName(m)), h("td", {}, prob(a.threshold?.[ev]?.[m])),
        ALERT.map(([k, , d, p]) => cell(byM[m][k], d, p)))))));
  }
  fill("alerts", out);
}

const ATTR = { age_band: "Age, years", sex: "Sex", asa: "ASA class", department: "Department", emergency: "Emergency or planned" };

function subgroups(W) {
  const H = HH(), fm = metrics.final_model, out = [
    h("p", {}, `AUROC ${H} minutes ahead within each patient group. Small groups have wide intervals; read them before reading the dots.`), legend()];
  const cw = W >= 760 ? (W - 32) / 2 : W;
  out.push(...missing(metrics.subgroups));
  for (const [ev, byH] of Object.entries(metrics.subgroups || {})) {
    out.push(block(ev, evHead(ev), h("div", { class: "pair" }, Object.entries(byH[H] || {}).map(([attr, groups]) =>
      dumbbell(ATTR[attr] || attr, groups.map((g) => ({ label: g.label, sub: `${fmt(g.n_pos)} of ${fmt(g.n)} times before an onset`, a: g.models[fm]?.auroc, b: g.models.baseline1?.auroc })),
        cw, [modelName(fm), MODEL.baseline1], fake(ev))))));
  }
  fill("subgroups", out.length > 2 ? out : empty("Subgroup results not computed.", RUN.evaluate));
}

// ---- the answer first: the final numbers in plain words, every one read from metrics.json
function summarySection() {
  const fm = metrics.final_model, H = HH(), all = flagged("overall")?.hypotension?.[H] || {}, bins = flagged("bins")?.hypotension?.[H] || [];
  const o = all[fm], b = all.baseline1;
  if (!o || !b) return fill("summary", empty("No hypotension results yet.", RUN.evaluate));
  const n = metrics.n?.hypotension || {}, nHyp = ill ? realN?.events : metrics.n?.hypoxemia?.events;
  const f1 = (v) => fmt(v, 1), f2 = (v) => fmt(v, 2), p0 = (v) => fmt(v * 100);
  const ci = (m, f) => (m ? `${f(m[1])}–${f(m[2])}` : "n/a");
  const tile = (label, k, f, unit, sub) => h("div", { class: "tile" },
    h("p", { class: "tile-label" }, label),
    h("p", { class: "tile-num" }, `${f(o[k]?.[0])}${unit}`),
    h("p", { class: "tile-vs" }, "Current MAP alone: ", h("strong", {}, `${f(b[k]?.[0])}${unit}`)),
    h("p", { class: "tile-ci" }, `95 % CI ${ci(o[k], f)} vs ${ci(b[k], f)}`),
    sub ? h("p", { class: "tile-sub" }, sub) : null);
  const more = o.event_recall?.[0] > b.event_recall?.[0], moreIsEarlier = (o.median_lead_min?.[0] > b.median_lead_min?.[0]) === more;
  const band = (bn) => {
    const a = bn.models[fm]?.auroc, z = bn.models.baseline1?.auroc;
    if (!a || !z) return null;
    // evaluate's paired bootstrap: the 95 % interval of the AUROC gain is above 0 (fallback: the intervals are apart)
    const clear = bn.models[fm].beats_baseline1?.auroc_ci ?? a[1] > z[2];
    return h("div", { class: clear ? "band" : "band unclear" },
      h("p", { class: "band-label" }, binLabel("hypotension", bn.label), " mmHg"),
      h("p", { class: "band-num" }, f2(a[0]), h("span", { class: "vs" }, ` vs ${f2(z[0])}`)),
      h("p", { class: "tile-ci" }, `95 % CI ${ci(a, f2)} vs ${ci(z, f2)}`),
      h("p", { class: "band-verdict" }, clear ? "Clear win: the 95 % interval of the gain is above zero."
        : a[0] > z[0] ? "Higher, but the gain's 95 % interval includes zero: not a clear win." : "Not better in this band."));
  };
  fill("summary",
    h("p", { class: "hero" }, metrics.split === "test" ? `Evaluated once on ${fmt(n.cases)} surgeries the model never saw.` : `Evaluated on ${fmt(n.cases)} validation surgeries (not the final test).`),
    h("p", { class: "lead" }, `${fmt(n.subjects)} patients, ${fmt(n.events)} hypotension events, ${fmt(n.times)} prediction times. Hypotension ${H} minutes ahead: our model against an alarm on the current MAP alone (baseline 1), each alarm threshold set on validation surgeries for at most ${fmt(metrics.alerts?.alarms_per_hour_target)} alarms per hour.`,
      exclFlagged ? " Times within 5 minutes after an event are left out (setting further down)." : null),
    h("div", { class: "tiles" },
      tile(`Ranks risk ${H} min ahead (AUROC, 0.5 = chance)`, "auroc", f2, ""),
      tile("Median warning before a drop", "median_lead_min", f1, " min", "Over the drops each one warned."),
      tile("False alarms per true alarm", "false_per_true_alarm", f2, "", `Lower is better. At ${f1(o.alarms_per_hour?.[0])} vs ${f1(b.alarms_per_hour?.[0])} alarms per hour.`),
      tile(`Drops with an alarm in the ${H} min before`, "event_recall", p0, " %",
        `${more ? "Our model" : "The current-MAP alarm"} catches more drops${moreIsEarlier ? " and warns earlier" : ", but warns later"}.`)),
    h("h3", {}, "While MAP still looks normal"),
    h("p", {}, `AUROC inside each band of the current MAP. Inside a narrow band the number alone says little, so this is where an earlier warning has to come from.`),
    h("div", { class: "bands" }, bins.map(band)),
    h("p", { class: "note" }, `Hypoxemia: only ${nHyp == null ? "a few" : fmt(nHyp)} events in these surgeries, so it is exploratory and no claim is made.`,
      ill ? h("span", { class: "ill-tag" }, " The hypoxemia charts further down show made-up illustrative numbers.") : null),
    h("p", { class: "claim" }, "What this shows: more time to act before the pressure falls, from signals already on the monitor. Retrospective and from one hospital (VitalDB); it does not show that patients do better."));
}

let cohort, metrics, index;

function renderAll() {
  overview(cohort);
  for (const id of ["summary", "headline", "metrics", "calibration", "alerts", "subgroups"])
    if (!metrics) fill(id, empty("Not computed yet.", RUN.evaluate));
  if (metrics) {
    const W = document.querySelector("#headline .body").clientWidth;
    summarySection();
    headline(W);
    metricsSection();
    calibration(W);
    alertsSection();
    subgroups(W);
  }
  if (!index?.length) fill("cases", empty("No surgeries exported yet.", RUN.export));
  else { if (!picker) caseShell(); drawCase(); }
}

// hypoxemia blocks only; with no metrics.json, a skeleton carries just enough for the sections to draw them
function useIllustrative() {
  realN = metrics?.n?.hypoxemia; realSplit = metrics?.split;
  metrics ||= { final_model: ill.final_model, models: ill.models, headline_horizon_min: ill.headline_horizon_min };
  for (const k of BLOCKS) metrics[k] = { ...metrics[k], hypoxemia: ill[k] };
  metrics.alerts = { ...metrics.alerts, threshold: { ...metrics.alerts?.threshold, hypoxemia: ill.alert_threshold } };
}

async function main() {
  const wantIll = q.get("illustrative") === "1";
  let illFile;
  [cohort, metrics, index, illFile, featured] = await Promise.all([get("cohort.json"), get("metrics.json"), get("cases/index.json"),
    wantIll ? get("illustrative_hypoxemia.json", "") : null, get("featured.json", "")]);
  featured ||= [];
  const status = document.getElementById("status");
  if (location.protocol === "file:")
    status.replaceChildren("Open this page through the local server so it can read the results: ",
      h("code", {}, "uv run python -m http.server -d ~/Desktop/hackathon 8000"), ", then http://localhost:8000/web/");
  else if (metrics)
    status.textContent = metrics.split === "test"
      ? "Results on the held-out test split, run once."
      : "Results on the validation split.";
  else status.textContent = "No model has been evaluated yet. Sections without results say what to run.";
  if (wantIll && illFile?.illustrative === true) { // the file must say so itself, or it is not shown
    ill = illFile;
    status.textContent = `Hypotension: ${metrics ? status.textContent.toLowerCase() : "no model has been evaluated yet; its sections say what to run."} Hypoxemia: made-up illustrative numbers, see the banner.`;
    useIllustrative();
    document.title += " (illustrative)";
    document.querySelector("header").prepend(h("div", { class: "ill-banner", role: "note" }, h("strong", {}, ILL),
      h("p", {}, "Every hypoxemia number, chart and table on this page is made up to show what the results could look like with more surgeries. No model produced them. Hypotension results and the surgeries below are real. ",
        h("a", { href: location.pathname }, "Show only real results"), ".")));
  } else if (wantIll) status.append(" ", empty("Illustrative hypoxemia numbers were asked for but not found.", RUN.illustrative));
  renderAll();
  if (hashCase()) document.getElementById("cases").scrollIntoView(); // a #case= link lands on the surgery
  let raf;
  addEventListener("resize", () => { cancelAnimationFrame(raf); raf = requestAnimationFrame(renderAll); });
}
main();
