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

let cohort, metrics, index;

function renderAll() {
  overview(cohort);
  for (const id of ["headline", "metrics", "calibration", "alerts", "subgroups"])
    if (!metrics) fill(id, empty("Not computed yet.", RUN.evaluate));
  if (!index) fill("cases", empty("No surgeries exported yet.", RUN.export));
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
