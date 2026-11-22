"use strict";
// Story page (PLAN.md "Live demo"). Every number shown comes from reports/web/metrics.json, cohort.json or the server's
// /api/health and /api/insights; none is typed in here. Hypotension only: hypoxemia is exploratory. The replay is live.html.

const q = new URLSearchParams(location.search);
// ?theme=light|dark overrides the default light theme (projectors, screenshots)
if (q.get("theme")) document.documentElement.dataset.theme = q.get("theme");
const DATA = "../reports/web/", API = "/api/";
const N_FEATURES = 131; // feature_names in data/processed/models/snapshot_hypotension_10.json (tests/test_story_page.py)
const N_LEAK = 13; // SPEC.md "Leakage checklist" items (tests/test_story_page.py)

// ---- builders, as in app.js (which runs its own page on load, so it is not loaded here)
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
const $ = (id) => document.getElementById(id);
const fmt = (v, d = 0) => (v == null ? "—" : v.toLocaleString("en", { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (v, d = 0) => (v == null ? "—" : `${fmt(v * 100, d)} %`); // no-break space: "%" never wraps alone
const f1 = (v) => fmt(v, 1), f2 = (v) => fmt(v, 2), p0 = (v) => fmt(v * 100);
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const bandName = (l) => (l.startsWith(">") ? `MAP above ${l.slice(1)}` : `MAP ${l.replace("-", "–")}`);
async function get(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

let metrics, cohort, health, insights;
const head = () => {
  const H = String(metrics?.headline_horizon_min), byM = metrics?.overall?.hypotension?.[H];
  return byM?.[metrics.final_model] && byM.baseline1 ? { H, o: byM[metrics.final_model], b: byM.baseline1 } : null;
};
const bands = () => { const x = head(), fm = metrics?.final_model; return x ? (metrics.bins?.hypotension?.[x.H] || []).filter((bn) => bn.models[fm]?.auroc && bn.models.baseline1?.auroc) : []; };
const clearWin = (bn) => !!bn.models[metrics.final_model].beats_baseline1?.auroc_ci; // paired-bootstrap 95 % CI of the gain > 0
const threshold = () => metrics?.alerts?.threshold?.hypotension?.[metrics.final_model];

// ---- 1. hero
function hero() {
  const el = $("hero-nums"), x = head(), n = metrics?.n?.hypotension;
  if (!x || !n) return el.replaceChildren(h("p", { class: "empty" }, "Results are not computed yet."));
  const { H, o, b } = x, bs = bands(), nClear = bs.filter(clearWin).length;
  const hn = (num, vs, label) => h("div", { class: "hn" }, h("p", { class: "hn-num" }, num), h("p", { class: "hn-vs" }, vs), h("p", { class: "hn-label" }, label));
  el.replaceChildren(
    hn(f2(o.auroc[0]), `vs ${f2(b.auroc[0])} for the current MAP alone`, `ranks risk ${H} min ahead (AUROC; 0.5 = coin flip)`),
    hn(`${f1(o.median_lead_min[0])} min`, `vs ${f1(b.median_lead_min[0])} min for the current-MAP alarm`, "median warning before a drop, over the drops warned"),
    bs.length ? hn(`${nClear} of ${bs.length}`, "MAP bands with a clear win", "while the MAP still looks normal: the gain's paired-bootstrap 95 % interval is above zero") : null,
    h("p", { class: "hn-foot" }, metrics.split === "test"
      ? `On ${fmt(n.cases)} surgeries (${fmt(n.subjects)} patients) the model never saw: the locked test split, evaluated once.`
      : `On ${fmt(n.cases)} validation surgeries, not the final test.`,
    o.event_recall[0] < b.event_recall[0] ? ` The current-MAP alarm catches more drops (${pct(b.event_recall[0])} vs ${pct(o.event_recall[0])}), but later: see the trade-off below.` : null));
}

// ---- 2. how it works
function nodes() {
  const thr = threshold();
  return {
    sig: ["Cleaned signals", "2 s grid, causal"], feat: [`${N_FEATURES} features`, "every 30 s"], xgb: ["XGBoost", "snapshot model"],
    cur: ["Current MAP", "and 5-min slope"], base: ["Trend baseline", "logistic regression"], fus: ["Fusion stacker", "logistic regression"],
    iso: ["Isotonic", "calibration"], alarm: ["Alarm", thr != null ? `10-min risk ≥ ${pct(thr)}` : "risk ≥ threshold"],
  };
}
const EDGES = [["sig", "feat"], ["sig", "cur"], ["feat", "xgb"], ["cur", "base"], ["xgb", "fus"], ["base", "fus"], ["fus", "iso"], ["iso", "alarm"]];
function pipeline() {
  const el = $("pipeline"), wide = (el.clientWidth || 1000) >= 760;
  const L = wide
    ? { w: 1200, h: 236, bw: 172, bh: 76, fs: [19, 15], pos: { sig: [0, 80], feat: [205, 10], cur: [205, 150], xgb: [410, 10], base: [410, 150], fus: [615, 80], iso: [820, 80], alarm: [1025, 80] } }
    : { w: 360, h: 552, bw: 168, bh: 64, fs: [16, 13], pos: { sig: [96, 0], feat: [0, 96], cur: [192, 96], xgb: [0, 192], base: [192, 192], fus: [96, 288], iso: [96, 384], alarm: [96, 480] } };
  const g = [s("defs", {}, s("marker", { id: "arr", viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
    s("path", { class: "arrowhead", d: "M0,0L10,5L0,10z" })))];
  for (const [a, b] of EDGES) {
    const [ax, ay] = L.pos[a], [bx, by] = L.pos[b];
    const d = wide
      ? (() => { const x1 = ax + L.bw, y1 = ay + L.bh / 2, x2 = bx - 2, y2 = by + L.bh / 2, m = (x1 + x2) / 2; return `M${x1},${y1}C${m},${y1} ${m},${y2} ${x2},${y2}`; })()
      : (() => { const x1 = ax + L.bw / 2, y1 = ay + L.bh, x2 = bx + L.bw / 2, y2 = by - 2, m = (y1 + y2) / 2; return `M${x1},${y1}C${x1},${m} ${x2},${m} ${x2},${y2}`; })();
    g.push(s("path", { class: "edge", d, "marker-end": "url(#arr)" }));
  }
  for (const [k, [t, sub]] of Object.entries(nodes())) {
    const [nx, ny] = L.pos[k], cx = nx + L.bw / 2, cy = ny + L.bh / 2;
    g.push(s("g", { class: k === "alarm" ? "node key" : "node" }, s("rect", { x: nx + 1, y: ny + 1, width: L.bw - 2, height: L.bh - 2, rx: 10 }),
      s("text", { x: cx, y: cy - 3, "text-anchor": "middle", style: `font-size:${L.fs[0]}px;font-weight:600` }, t),
      s("text", { class: "n-sub", x: cx, y: cy + L.fs[1] + 3, "text-anchor": "middle", style: `font-size:${L.fs[1]}px` }, sub)));
  }
  el.replaceChildren(s("svg", { viewBox: `0 0 ${L.w} ${L.h}`, role: "img", "aria-label": "Pipeline: cleaned monitor signals feed both the feature model (XGBoost) and a trend baseline on the current MAP; a fusion stacker combines them, isotonic calibration turns the score into a probability, and the alarm fires at the threshold." }, g));
}
function steps() {
  const hs = metrics?.horizons_min, target = metrics?.alerts?.alarms_per_hour_target, thr = threshold();
  const list = [
    ["Cleaned signals", "Arterial pressure, heart rate, SpO2, ventilation and drug infusions on a 2 s grid, with artifacts such as line flushes removed using earlier samples only."],
    [`${N_FEATURES} features`, "Every 30 s: levels, lows and slopes over the last 1 to 10 minutes, beat-by-beat arterial waveform features, drug rates, the phase of surgery and pre-op fields."],
    ["XGBoost", `A gradient-boosted tree model for each horizon${hs ? ` (${hs.join(", ")} min)` : ""} turns the features into a risk score.`],
    ["Trend baseline", "A logistic regression on the current MAP and its 5-minute slope: roughly what a careful look at the monitor gives."],
    ["Fusion stacker", "Combines both scores with the current MAP and the phase of surgery, and learned only from out-of-fold predictions."],
    ["Isotonic calibration", "Turns the fused score into a probability that matches the rate seen on validation surgeries."],
    ["Alarm", `Fires when the 10-min risk reaches ${thr != null ? pct(thr) : "the threshold"}, set on validation surgeries${target ? ` for at most ${fmt(target)} alarms per hour` : ""}, then stays quiet for 5 min unless the risk climbs 10 points more.`],
  ];
  $("steps").replaceChildren(...list.map(([t, d]) => h("li", {}, h("strong", {}, t), d)));
}

// ---- 3. kept honest
function honest() {
  const sp = cohort?.splits, n = metrics?.n?.hypotension, name = { train: "Training", val: "Validation", test: "Test" };
  $("honest-body").replaceChildren(
    h("div", { class: "card" }, h("h3", {}, "Split by patient"),
      h("p", {}, "All of a patient's surgeries sit in one split, so the test is on people the model has never seen."),
      sp ? ["train", "val", "test"].map((k) => h("p", { class: "split-row" }, h("span", {}, name[k]),
        h("span", {}, `${fmt(sp[k]?.n_subjects)} patients · ${fmt(sp[k]?.n_cases)} surgeries`))) : h("p", { class: "note" }, "Split counts not exported yet."),
      n ? h("p", { class: "note" }, `Counts over the whole sampled cohort; ${fmt(n.cases)} test surgeries (${fmt(n.subjects)} patients) had a scored prediction time.`) : null),
    h("div", { class: "card" }, h("h3", {}, `${N_LEAK} leakage tests`),
      h("p", {}, "Each checklist item is an automated test; no result was reported until all passed. Among them:"),
      h("ul", {}, ["Features at a time t are unchanged when every sample after t is deleted.",
        "The label window never overlaps the input window.", "The alarm threshold is chosen on validation, never on test.",
        "The stacker trains only on out-of-fold predictions."].map((t) => h("li", {}, t))),
      h("p", { class: "note" }, "The full list is in SPEC.md.")),
    h("div", { class: "card" }, h("h3", {}, "One locked test run"),
      h("p", {}, "The test split was scored once, at the end, and a lock file stops that run from happening again. Nothing was tuned afterwards."),
      h("p", {}, "The live monitor runs the saved models again and reproduces those test predictions exactly; an automated test checks it.")));
}

// ---- 4. results (metrics.json; the verdicts are evaluate's paired bootstrap, as on the results page)
function results() {
  const el = $("results-body"), x = head(), n = metrics?.n?.hypotension;
  if (!x || !n) return el.replaceChildren(h("p", { class: "empty" }, "No results yet. Run ", h("code", {}, "uv run python -m orew.evaluate --split val"), "."));
  const { H, o, b } = x;
  const tile = (label, k, f, unit = "") => h("div", { class: "tile" }, h("p", { class: "tile-label" }, label),
    h("p", { class: "tile-num" }, f(o[k][0]) + unit), h("p", { class: "tile-vs" }, "Current MAP alone: ", h("strong", {}, f(b[k][0]) + unit)),
    h("p", { class: "tile-ci" }, `95 % CI ${f(o[k][1])}–${f(o[k][2])} vs ${f(b[k][1])}–${f(b[k][2])}`));
  el.replaceChildren(
    h("p", { class: "sub" }, `${metrics.split === "test" ? "Evaluated once" : "Validation split, not the final test"}: ${fmt(n.cases)} surgeries, ${fmt(n.subjects)} patients, ${fmt(n.events)} hypotension events, ${fmt(n.times)} prediction times. Hypotension ${H} min ahead; 95 % CIs from ${fmt(metrics.bootstrap)} patient-bootstrap resamples.`),
    h("div", { class: "tiles" }, tile(`Ranks risk ${H} min ahead (AUROC, 0.5 = coin flip)`, "auroc", f2), tile("Precision-recall (AUPRC)", "auprc", f2),
      tile("Sensitivity at 90 % specificity", "sens_at_90spec", p0, " %")),
    h("div", { class: "card" },
      h("h3", {}, "While MAP still looks normal"),
      h("p", {}, `Inside a narrow band of the current MAP, the number alone says little: that is where an earlier warning has to come from. AUROC ${H} min ahead in each band, with 95 % CIs. "Clear win": the paired-bootstrap 95 % interval of the gain is above zero.`),
      h("p", { class: "legend" }, h("span", { class: "key key-final", "aria-hidden": "true" }), "Our model  ",
        h("span", { class: "key key-base", "aria-hidden": "true" }), "Current MAP alone"),
      h("div", { class: "band-chart", id: "band-chart" })),
    h("div", { class: "card" }, h("h3", {}, "The alarm trade-off"), tradeoff(o, b, H)));
  bandChart();
}

function bandChart() {
  const el = $("band-chart"), x0 = head();
  if (!el || !x0) return;
  const fm = metrics.final_model, bins = bands();
  const W = el.clientWidth || 900, narrow = W < 640, LW = narrow ? 0 : 200, VW = narrow ? 0 : 120, RH = narrow ? 76 : 64, TOP = 30;
  const vals = bins.flatMap((bn) => [...bn.models[fm].auroc, ...bn.models.baseline1.auroc]);
  const lo = Math.min(0.45, Math.floor(Math.min(...vals) * 20) / 20), hi = Math.ceil(Math.max(...vals) * 20) / 20;
  const x = lin(lo, hi, LW + 10, W - VW - 44), H = TOP + bins.length * RH + 26, g = [];
  for (let v = Math.ceil(lo * 10) / 10; v <= hi + 1e-9; v += 0.1)
    g.push(s("line", { class: "grid", x1: x(v), x2: x(v), y1: TOP - 6, y2: H - 22 }), s("text", { x: x(v), y: H - 6, "text-anchor": "middle" }, fmt(v, 1)));
  g.push(s("line", { class: "coin", x1: x(0.5), x2: x(0.5), y1: TOP - 10, y2: H - 22 }),
    s("text", { class: "coin-label", x: x(0.5) + 6, y: TOP - 14 }, "0.5 = coin flip"));
  bins.forEach((bn, i) => {
    const y = TOP + i * RH, a = bn.models[fm].auroc, z = bn.models.baseline1.auroc, clear = clearWin(bn);
    const verdict = clear ? "clear win" : a[0] > z[0] ? "not proven" : "not better";
    const label = `${bandName(bn.label)} mmHg`, sub = `${fmt(bn.n_pos)} of ${fmt(bn.n)} times before a drop`;
    if (narrow) g.push(s("text", { class: "row-label", x: 0, y: y + 14, style: "font-size:15px" }, label, s("tspan", { class: clear ? "verdict-win" : "verdict-no" }, `  · ${verdict}`)));
    else g.push(s("text", { class: "row-label", x: 0, y: y + 24, style: "font-size:17px" }, label), s("text", { x: 0, y: y + 44, style: "font-size:13px" }, sub),
      s("text", { class: clear ? "verdict-win" : "verdict-no", x: W - VW + 8, y: y + 34, style: "font-size:17px" }, verdict));
    for (const [m, dy, cls] of [[a, narrow ? 34 : 22, "mk-final"], [z, narrow ? 56 : 44, "mk-base"]])
      g.push(s("g", { class: cls }, s("title", {}, `${cls === "mk-final" ? "Our model" : "Current MAP alone"}: ${fmt(m[0], 3)} (95 % CI ${fmt(m[1], 3)}–${fmt(m[2], 3)})`),
        s("line", { x1: x(m[1]), x2: x(m[2]), y1: y + dy, y2: y + dy }), s("line", { x1: x(m[1]), x2: x(m[1]), y1: y + dy - 5, y2: y + dy + 5 }),
        s("line", { x1: x(m[2]), x2: x(m[2]), y1: y + dy - 5, y2: y + dy + 5 }), s("circle", { cx: x(m[0]), cy: y + dy, r: 6 })),
      s("text", { x: x(m[2]) + 8, y: y + dy + 5, class: cls === "mk-final" ? "val-final" : "val-base" }, f2(m[0])));
  });
  const table = h("table", { class: "vh" }, h("caption", {}, "AUROC by current MAP band"),
    bins.map((bn) => h("tr", {}, h("th", {}, bandName(bn.label)), h("td", {}, `Our model ${f2(bn.models[fm].auroc[0])}`),
      h("td", {}, `Current MAP ${f2(bn.models.baseline1.auroc[0])}`), h("td", {}, clearWin(bn) ? "clear win" : "not proven"))));
  el.replaceChildren(s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, g), table);
}

function tradeoff(o, b, H) {
  const rows = [
    ["event_recall", "Drops caught (recall)", `share of drops with an alarm in the ${H} min before`, (v) => `${p0(v)} %`, true],
    ["false_per_true_alarm", "False alarms per true alarm", "lower is better", f2, false],
    ["median_lead_min", "Median warning", "before the drop, over the drops warned", (v) => `${f1(v)} min`, true],
  ];
  const better = (k, higher) => (higher ? o[k][0] > b[k][0] : o[k][0] < b[k][0]);
  const blocks = rows.map(([k, title, note, f, higher]) => {
    const mx = Math.max(o[k][0], b[k][0]) || 1, w = (v) => `${((v / mx) * 58).toFixed(1)}%`;
    const bar = (who, v, base) => h("div", { class: "bar-row" }, h("span", { class: "who-l" }, who),
      h("div", { class: base ? "bar base" : "bar" }, h("span", { style: `width:${w(v)}` }), h("b", { style: `left:${w(v)}` }, f(v))));
    return h("div", {}, h("h4", {}, title), h("p", { class: "better" }, `${note} · ${better(k, higher) ? "ours is better" : "the current-MAP alarm is better"}`),
      bar("Our model", o[k][0]), bar("Current MAP alone", b[k][0], true));
  });
  const v = (k, f) => `${f(o[k][0])} vs ${f(b[k][0])}`;
  const rec = better("event_recall", true), lead = better("median_lead_min", true), fa = better("false_per_true_alarm", false);
  const a = metrics.alerts || {};
  return [h("div", { class: "trade" }, blocks),
    h("p", { class: "trade-verdict" }, rec ? `Our model catches more drops (${v("event_recall", pct)})` : `The current-MAP alarm catches more drops (${pct(b.event_recall[0])} vs ${pct(o.event_recall[0])})`,
      `. Ours warns ${lead ? "earlier" : "later"} (median ${v("median_lead_min", f1)} min),`,
      ` with ${fa ? "fewer" : "more"} false alarms per true alarm (${v("false_per_true_alarm", f2)}).`),
    h("p", { class: "note" }, `Both alarms fire at about the same rate (${v("alarms_per_hour", f1)} per hour): each threshold was set on ${a.chosen_on === "val" ? "validation" : a.chosen_on} surgeries for at most ${fmt(a.alarms_per_hour_target)} alarms per hour. A drop is warned if an alarm fired in the ${H} min before it; repeat alarms within 5 min count once.`)];
}

// ---- 5. Tiger Data (/api/insights: orew.tiger runs these SQL queries on Tiger Cloud when TIGER_DATABASE_URL is set)
function tiger() {
  const el = $("tiger-body"), src = health?.data_source;
  const intro = h("p", { class: "sub" }, "Tiger Cloud is managed TimescaleDB: Postgres for time series. With a database connected, the test surgeries' vital signs and the model's predictions, alarms and events sit in hypertables, and the queries below run on them live.");
  const srcLine = health ? h("p", { class: "src-line" }, h("span", { class: src === "tiger" ? "tag on" : "tag" }, src === "tiger" ? "Tiger Cloud" : "Local files"),
    ` The live monitor reads its vital signs from ${src === "tiger" ? "the Tiger Cloud grid hypertable" : "local Parquet files"}.`) : null;
  if (!insights?.available) return el.replaceChildren(intro, h("div", { class: "card not-connected" },
    h("p", { class: "nc-head" }, h("span", { class: "tag" }, "Not connected"), " ", insights ? insights.reason || "No database." : "The demo server is not running."),
    h("p", { class: "note" }, "Set TIGER_DATABASE_URL in the environment or the gitignored .env, load the data with ", h("code", {}, "uv run python -m orew.tiger load"),
      " and restart the server. Until then the live monitor runs from local files."), srcLine));
  el.replaceChildren(intro, srcLine, h("div", { class: "queries" }, (insights.queries || []).map(queryCard)));
}
function queryCard(qr) {
  const cols = qr.columns || [], rows = (qr.rows || []).map((r) => (Array.isArray(r) ? r : cols.map((c) => r[c])));
  const num = cols.map((_, i) => rows.length > 0 && rows.every((r) => r[i] == null || typeof r[i] === "number"));
  // the bar shows the first numeric column after the label column; one row needs no bar
  const bi = rows.length > 1 ? num.indexOf(true, 1) : -1, mx = bi >= 0 ? Math.max(...rows.map((r) => Math.abs(r[bi] ?? 0))) || 1 : 1;
  // a numeric column keeps one number of decimals (up to 3); ids get no thousands separator
  const dec = cols.map((_, i) => Math.min(3, Math.max(0, ...rows.map((r) => (typeof r[i] === "number" ? (String(r[i]).split(".")[1] || "").length : 0)))));
  const cell = (v, i) => (v == null ? "—" : typeof v === "number"
    ? v.toLocaleString("en", { minimumFractionDigits: dec[i], maximumFractionDigits: dec[i], useGrouping: !/id$/.test(cols[i]) }) : String(v));
  const shown = rows.slice(0, 20);
  // the bar cell sits right after the column it draws
  const barCell = (r) => h("td", { class: "bar", "aria-hidden": "true" }, h("span", { style: `width:${((Math.abs(r[bi] ?? 0) / mx) * 100).toFixed(1)}%` }));
  const thead = h("thead", {}, h("tr", {}, cols.map((c, i) => [h("th", { scope: "col", class: i === bi ? "barred" : null }, c.replaceAll("_", " ")), i === bi ? h("td", { class: "bar" }) : null])));
  const tbody = h("tbody", {}, shown.map((r) => h("tr", {}, r.map((v, i) => [
    i ? h("td", { class: i === bi ? "barred" : null }, cell(v, i)) : h("th", { scope: "row" }, cell(v, i)), i === bi ? barCell(r) : null]))));
  return h("article", { class: "card query" }, h("h3", {}, qr.title), h("p", {}, qr.plain),
    rows.length ? h("div", { class: "scroll" }, h("table", {}, thead, tbody)) : h("p", { class: "note" }, "No rows."),
    rows.length > shown.length ? h("p", { class: "note" }, `${rows.length - shown.length} more rows not shown.`) : null,
    h("details", {}, h("summary", {}, "Show the SQL"), h("pre", {}, h("code", {}, qr.sql || ""))));
}

// ---- 6. the two limitations that quote results
function limits() {
  const x = head();
  if (!x) return;
  const { o, b } = x, unclear = bands().filter((bn) => !clearWin(bn)), fm = metrics.final_model;
  const t = $("lim-trade");
  t.textContent = `Trade-off: the alarm on the current MAP catches ${b.event_recall[0] > o.event_recall[0] ? "more" : "fewer"} drops (${pct(b.event_recall[0])} against ${pct(o.event_recall[0])}), but ${b.median_lead_min[0] < o.median_lead_min[0] ? "later" : "earlier"}; the model raises ${f2(o.false_per_true_alarm[0])} false alarms per true alarm.`;
  t.hidden = false;
  if (unclear.length) {
    const l = $("lim-bands");
    l.textContent = `${unclear.map((bn) => bandName(bn.label)).join(", ")} mmHg: ${unclear.every((bn) => bn.models[fm].auroc[0] > bn.models.baseline1.auroc[0]) ? "a higher AUROC, but not" : "not"} a clear win.`;
    l.hidden = false;
  }
}

async function main() {
  [metrics, cohort, health, insights] = await Promise.all([get(`${DATA}metrics.json`), get(`${DATA}cohort.json`), get(`${API}health`), get(`${API}insights`)]);
  hero(); pipeline(); steps(); honest(); results(); tiger(); limits();
  if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView({ behavior: "instant" }); // the anchor moved while sections filled
  let rz;
  addEventListener("resize", () => { cancelAnimationFrame(rz); rz = requestAnimationFrame(() => { pipeline(); bandChart(); }); });
}
main();
