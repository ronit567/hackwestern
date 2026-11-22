"use strict";
// Live patient monitor (PLAN.md "Live demo"): replays one unseen test surgery from /api/replay/{id}. Every number shown
// comes from the stream (live model output; onsets from events.parquet), /api/surgeries, featured.json text or a case
// file; none is typed in here. Hypotension only: hypoxemia is exploratory. Nothing here looks past the replay clock,
// except the presenter's "skip" button, which uses the case file's drop times only to choose where to start.
// URL: ?case=5566&start=<sec>&speed=120&autoplay=1 (&pause=<sec>: stop the clock there, for screenshots)

const q = new URLSearchParams(location.search);
const DATA = "../reports/web/", API = "/api/";
const MAP_THR = 65, SPO2_THR = 90; // the event definitions: hypotension MAP < 65 mmHg, hypoxemia SpO2 < 90 %
const AHEAD_S = 600, ALARM_SHOW_S = 300, SPARK_S = 900, TICK_S = 30; // 10-min horizon; banner 5 min; sparkline 15 min
const TREND_EPS = 0.01; // "rising"/"falling" when the 10-min risk moved more than 1 point in 5 min, else "steady"
const SPEEDS = [30, 60, 120, 300];
const clampSpeed = (v) => Math.min(600, Math.max(1, v)); // the server's clamp
const LEAD = { snapshot: "vital-sign pattern", trend: "MAP trend", current_value: "current MAP", phase: "surgical phase" };
const VITALS = [
  { k: "hr", label: "Heart rate", unit: "bpm" },
  { k: "spo2", label: "SpO₂", unit: "%", low: SPO2_THR },
  { k: "bt", label: "Temp", unit: "°C", d: 1 },
  { k: "sbp", label: "SBP", unit: "mmHg" },
  { k: "map", label: "MAP", unit: "mmHg", low: MAP_THR, main: true },
  { k: "dbp", label: "DBP", unit: "mmHg" },
  { k: "rr", label: "Resp rate", unit: "/min" },
  { k: "etco2", label: "EtCO₂", unit: "mmHg" },
];

// ---- builders, as in app.js
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
const pc = (v) => (v == null ? "—" : v < 0.005 ? "<1%" : `${Math.round(v * 100)}%`);
const hm = (sec) => { if (sec == null) return "—"; const m = Math.floor(Math.max(0, sec) / 60); return `${Math.floor(m / 60)}:${String(m % 60).padStart(2, "0")}`; };
const dur = (min) => { const m = Math.round(min); return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")}`; };
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
const median = (a) => { if (!a.length) return null; const b = [...a].sort((p, r) => p - r), m = b.length >> 1; return b.length % 2 ? b[m] : (b[m - 1] + b[m]) / 2; };
const drops = (n) => `${n} ${n === 1 ? "drop" : "drops"}`;
const measure = document.createElement("canvas").getContext("2d"); // label widths for the chart's bracket rows
// [[t, v | null]] -> path; a null lifts the pen
function pathD(pts, x, y) {
  let d = "", pen = false;
  for (const [t, v] of pts) {
    if (v == null) { pen = false; continue; }
    d += `${pen ? "L" : "M"}${x(t).toFixed(1)},${y(v).toFixed(1)}`;
    pen = true;
  }
  return d;
}
async function get(url) {
  try {
    const r = await fetch(url);
    return r.ok ? await r.json() : null;
  } catch {
    return null;
  }
}

// ---- state
let surgeries = [], index = [], health = null, R = null, es = null, raf = 0;
let speed = clampSpeed(+q.get("speed") || 60);
let pauseAt = q.has("pause") ? +q.get("pause") : null; // stop after the tick at this time (its alarm/onset included)
const fresh = (start) => ({ from: start, clock: null, vit: [], ticks: [], alarms: [], onsets: [], end: null, rows: new Map(), status: "" });
const playing = () => es != null;
const paint = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };
const info = (id) => surgeries.find((x) => x.caseid === id);
const demo = (id) => (R?.id === id && R.meta?.demographics) || index.find((c) => c.caseid === id);
const who = (d) => (d ? [d.age != null && `${fmt(d.age)} y`, d.sex].filter(Boolean).join(" ") : "");
const short = (reason) => reason?.split(/[:;,]/)[0].trim(); // "Honest failure: ..." -> "Honest failure"

// ---- 1. header tabs
function tabs() {
  const el = $("tabs");
  if (!surgeries.length) return el.replaceChildren();
  const feat = surgeries.filter((x) => x.featured), rest = surgeries.filter((x) => !x.featured);
  const sel = h("select", { id: "more", "aria-label": "More unseen test surgeries" }, h("option", { value: "" }, `More… (${rest.length})`),
    rest.map((x) => h("option", { value: x.caseid }, `Surgery ${x.caseid} · ${dur(x.duration_min)} · ${drops(x.n_events)}`)));
  sel.onchange = () => sel.value && choose(+sel.value, true);
  el.replaceChildren(...feat.map((x) => {
    const b = h("button", { type: "button", class: "tab", "data-case": x.caseid, "aria-pressed": "false", title: x.reason },
      h("span", { class: "tab-id" }, `Surgery ${x.caseid}`, h("span", { "data-who": x.caseid })),
      h("span", { class: "tab-label" }, short(x.reason) || `Surgery ${x.caseid}`),
      h("span", { class: "tab-foot" }, h("span", { class: `badge${x.n_events ? "" : " none"}` }, drops(x.n_events)), h("span", { class: "dept", "data-dept": x.caseid })));
    b.onclick = () => choose(x.caseid, true);
    return b;
  }), h("div", { class: "tab more", id: "more-tab" }, h("span", { class: "tab-id" }, "Other test surgeries", h("span", { class: "who", id: "more-who" })), sel));
}
function tabState() {
  for (const b of document.querySelectorAll(".tab[data-case]")) b.setAttribute("aria-pressed", String(+b.dataset.case === R?.id));
  for (const w of document.querySelectorAll("[data-who]")) w.textContent = who(demo(+w.dataset.who)) ? ` · ${who(demo(+w.dataset.who))}` : "";
  for (const w of document.querySelectorAll("[data-dept]")) w.textContent = demo(+w.dataset.dept)?.department || "";
  const sel = $("more"), other = R && !info(R.id)?.featured;
  if (!sel) return;
  sel.value = other ? String(R.id) : "";
  $("more-tab").classList.toggle("on", !!other);
  $("more-who").textContent = other ? ` · ${R.id}${who(demo(R.id)) ? ` · ${who(demo(R.id))}` : ""}${demo(R.id)?.department ? ` · ${demo(R.id).department}` : ""}` : "";
}

// ---- stream
function choose(id, play, start = 0) {
  closeStream();
  R = { id, file: undefined, meta: null, ...fresh(start) };
  $("rows").replaceChildren();
  // the case file gives the drop times for the presenter's skip and each drop's end for the shading; never sent to the model
  if (index.some((c) => c.caseid === id)) get(`${DATA}cases/${id}.json`).then((f) => { if (R.id === id) { R.file = f; paint(); } });
  else R.file = null;
  if (play) openStream(start, false);
  else R.status = "Press Play or the space bar to start.";
  paint();
}
function closeStream() { if (es) { es.close(); es = null; } }

// pause / seek = close the stream and reopen it at start_sec (PLAN.md). keep: resume, so vitals/ticks/alarms are deduped
function openStream(start, keep) {
  closeStream();
  if (!keep) { Object.assign(R, fresh(start)); $("rows").replaceChildren(); }
  const src = new EventSource(`${API}replay/${R.id}?speed=${speed}&start_sec=${Math.max(0, Math.round(start))}`);
  es = src;
  R.status = "Computing the replay… (a surgery's first replay takes a few seconds)";
  const on = (ev, f) => src.addEventListener(ev, (e) => { if (es !== src) return; f(JSON.parse(e.data)); paint(); });
  on("meta", (m) => { R.meta = m; R.status = ""; });
  on("tick", tick);
  on("alarm", alarm);
  on("onset", (o) => { if (!R.onsets.some((x) => x.t_sec === o.t_sec)) R.onsets.push(o); });
  on("end", (e) => { R.end = e.summary; R.status = "End of the surgery."; closeStream(); });
  src.onerror = () => { if (es !== src) return; closeStream(); R.status = "The replay stream stopped. Press Play to resume."; paint(); };
  paint();
}

function tick(k) {
  if (pauseAt != null && k.t_sec > pauseAt) { pauseAt = null; closeStream(); R.status = "Paused."; return; }
  const lastV = R.vit.at(-1)?.t_sec ?? -1;
  for (const v of k.vitals || []) if (v.t_sec > lastV) R.vit.push(v);
  if (R.ticks.at(-1)?.t_sec >= k.t_sec) return;
  R.ticks.push(k);
  R.clock = k.t_sec;
  const m = Math.floor(k.t_sec / 60), row = R.rows.get(m);
  if (!row?.alarm) putRow(m, k, null); // a minute's row shows its latest tick, or the tick its alarm fired at
}
function alarm(a) {
  if (R.alarms.some((x) => x.t_sec === a.t_sec)) return;
  R.alarms.push(a);
  const k = R.ticks.find((x) => x.t_sec === a.t_sec);
  if (k) putRow(Math.floor(a.t_sec / 60), k, a);
}

// ---- 3. minute-by-minute table (rows updated in place; newest at the bottom)
function putRow(m, k, a) {
  const thr = R.meta?.threshold, hot = k.p10 != null && thr != null && k.p10 >= thr;
  const tr = h("tr", { class: [hot && "hot", a && "alarm"].filter(Boolean).join(" ") || null },
    h("td", {}, hm(m * 60)),
    h("td", { class: k.map_now != null && k.map_now < MAP_THR ? "low" : null }, fmt(k.map_now)),
    h("td", {}, pc(k.p_baseline1_10)), h("td", {}, pc(k.p_snapshot_10)), h("td", { class: "fus" }, pc(k.p10)),
    h("td", { class: "lead" }, k.scored ? (LEAD[k.lead] ?? "—") : h("span", { class: "ns" }, "not scored")),
    h("td", { class: "al" }, a ? "⚠ ALARM" : ""));
  const old = R.rows.get(m)?.tr;
  if (old) old.replaceWith(tr);
  else $("rows").append(tr);
  R.rows.set(m, { tr, alarm: a });
  R.scroll = true; // keep the newest row in view, after the layout of this frame (draw)
}

// ---- controls
function toggle() {
  if (!R) return;
  if (playing()) { closeStream(); R.status = "Paused."; }
  else if (R.end || R.clock == null) openStream(R.end ? 0 : R.from, false);
  else openStream(R.clock, true); // resume from the last tick itself: its alarm/onset may not have arrived (deduped)
  paint();
}
function step() {
  if (!R || R.end) return;
  if (R.clock == null) { pauseAt = R.from; return openStream(R.from, false); }
  pauseAt = R.clock + TICK_S;
  openStream(R.clock, true);
}
function seek(t, play) {
  if (!R) return;
  pauseAt = play || playing() ? null : t; // seeking while paused shows that moment and stays paused
  openStream(t, false);
}
function setSpeed(v) {
  speed = v;
  if (playing()) openStream(R.clock ?? R.from, R.clock != null);
  paint();
}
const total = () => R?.meta?.duration_s ?? (R ? info(R.id)?.duration_min * 60 : null);
// presenter shortcut: 10 min before the next drop after the clock, on the 30 s grid (case file; the model never sees it).
// Chosen by onset, so a drop under 10 min away rewinds to its lead-up instead of being skipped.
function nextDrop() {
  const now = R?.clock ?? R?.from ?? 0;
  const on = (R?.file?.events || []).filter((e) => e.type === "hypotension" && e.onset_min * 60 > now).map((e) => e.onset_min * 60);
  return on.length ? Math.max(0, Math.floor((Math.min(...on) - AHEAD_S) / TICK_S) * TICK_S) : null;
}
// a drop is shaded from its onset to its end (case file), or until MAP is back at 65 when the surgery has no case file
function dropEnd(o) {
  const e = R.file?.events?.find((e) => e.type === "hypotension" && Math.abs(e.onset_min * 60 - o.t_sec) < 2);
  if (e) return e.end_min * 60;
  return R.vit.find((v) => v.t_sec > o.t_sec + 60 && v.map != null && v.map >= MAP_THR)?.t_sec ?? Infinity;
}
const activeDrop = () => R?.clock != null && R.onsets.find((o) => o.t_sec <= R.clock && R.clock < dropEnd(o));
const isTrue = (a) => R.onsets.find((o) => o.t_sec > a.t_sec && o.t_sec <= a.t_sec + AHEAD_S);
const settled = (a) => R.end || R.clock >= a.t_sec + AHEAD_S;
const lastTick = () => R?.ticks.at(-1);
function trend() { // fusion p10 now vs 5 min ago
  const k = lastTick(), b = R?.ticks.find((x) => x.t_sec === k?.t_sec - ALARM_SHOW_S);
  if (k?.p10 == null || b?.p10 == null) return null;
  const d = k.p10 - b.p10;
  return d > TREND_EPS ? "rising" : d < -TREND_EPS ? "falling" : "steady";
}

// ---- draw
function draw() {
  if (R?.scroll) { R.scroll = false; requestAnimationFrame(() => { const w = $("tbl-wrap"); w.scrollTop = w.scrollHeight; }); }
  tabState();
  vitals();
  banner();
  chart();
  scores();
  bar();
}

function vitals() {
  const now = R?.clock;
  const last = (k) => { // the monitor's number: the latest value in the last minute
    if (now == null) return null;
    for (let i = R.vit.length - 1; i >= 0 && R.vit[i].t_sec > now - 60; i--) if (R.vit[i][k] != null) return R.vit[i][k];
    return null;
  };
  const recent = now == null ? [] : R.vit.filter((v) => v.t_sec > now - SPARK_S);
  $("vitals").replaceChildren(...VITALS.map((V) => {
    const v = last(V.k), alert = v != null && V.low != null && v < V.low;
    const pts = recent.map((r) => [r.t_sec, r[V.k]]).filter((p) => p[1] != null);
    let spark = null;
    if (pts.length > 1) {
      const lo = Math.min(...pts.map((p) => p[1])), hi = Math.max(...pts.map((p) => p[1])), pad = Math.max(1, (hi - lo) * 0.1);
      spark = s("svg", { class: "spark", viewBox: "0 0 100 24", preserveAspectRatio: "none", "aria-hidden": "true" },
        s("path", { d: pathD(recent.map((r) => [r.t_sec, r[V.k]]), lin(now - SPARK_S, now, 0, 100), lin(lo - pad, hi + pad, 23, 1)) }));
    }
    return h("div", { class: `tile${V.main ? " key" : ""}${alert ? " alert" : ""}` },
      h("p", { class: "t-label" }, V.label),
      h("p", { class: "t-num" }, fmt(v, V.d || 0)),
      h("p", { class: "t-unit" }, alert ? `${V.unit} · below ${V.low}` : V.unit),
      spark || h("div", { class: "spark" }));
  }));
}

function banner() {
  const el = $("banner");
  if (!R) {
    el.className = "banner calm";
    return el.replaceChildren(h("p", { class: "b-title" }, "Live monitor"), h("p", { class: "b-line" }, surgeries.length ? "Choose a surgery above."
      : "The live monitor needs the demo server: run uv run python -m orew.app --port 8000 and open /web/live.html on it."));
  }
  const k = lastTick(), a = R.alarms.at(-1), thr = R.meta?.threshold, tr = trend(), drop = activeDrop();
  const sep = () => h("span", { class: "b-sep" }, "|");
  const mapNow = `MAP now ${fmt(k?.map_now)}`;
  const icon = (d) => s("svg", { class: "b-icon", viewBox: "0 0 24 24", "aria-hidden": "true" }, s("path", { d }));
  const WARN = "M12 3 2 21h20zM11 10h2v5h-2zm0 6.5h2v2h-2z", PULSE = "M2 12h4l2-5 4 10 3-7 2 2h5";
  if (drop) {
    el.className = "banner drop";
    return el.replaceChildren(h("div", { class: "b-head" }, icon(WARN), h("p", { class: "b-title" }, `MAP below 65 since ${hm(drop.t_sec)}`)),
      h("p", { class: "b-line" }, drop.warned ? h("strong", {}, `Warned ${fmt(drop.lead_min, 1)} min earlier by the alarm at ${hm(drop.alarm_t_sec)}`)
        : h("strong", {}, drop.warnable ? "Missed: no alarm in the 10 min before" : "Before the first prediction: not scored"),
      sep(), mapNow, sep(), "risk not scored during a drop"));
  }
  if (a && R.clock - a.t_sec <= ALARM_SHOW_S && !R.end) {
    el.className = "banner alarm";
    return el.replaceChildren(h("div", { class: "b-head" }, icon(WARN), h("p", { class: "b-title" }, "Hypotension alarm"), h("span", { class: "b-at" }, `fired at ${hm(a.t_sec)}`)),
      h("p", { class: "b-line" }, h("strong", {}, k?.p10 != null ? `Risk ${pc(k.p10)} in the next 10 min` : `Risk ${pc(a.p10)} at the alarm`),
        sep(), mapNow, tr ? [sep(), `Risk trend ${tr}`] : null),
      a.drivers?.length ? h("p", { class: "b-why" }, h("span", {}, "Why:"), a.drivers.slice(0, 3).map((d) => h("span", { class: "chip" }, d))) : null);
  }
  el.className = "banner calm";
  const first = R.meta?.first_pred_sec;
  const riskTxt = k?.p10 != null ? `Risk ${pc(k.p10)} in the next 10 min`
    : R.clock == null ? "Not started" : first != null && R.clock < first ? `First prediction at ${hm(first)}` : "Risk not scored at this moment";
  el.replaceChildren(h("div", { class: "b-head" }, icon(PULSE), h("p", { class: "b-title" }, "Monitoring")),
    h("p", { class: "b-line" }, h("strong", {}, riskTxt), k ? [sep(), mapNow] : null, tr ? [sep(), `Risk trend ${tr}`] : null),
    h("p", { class: "b-why" }, thr != null ? `An alarm fires when the 10-min risk reaches ${pc(thr)} (threshold chosen on validation surgeries), then stays quiet for 5 min unless the risk climbs further.` : " "));
}

function chart() {
  const box = $("chart"), W = Math.max(260, box.clientWidth), H = Math.max(120, box.clientHeight);
  const ML = 38, MR = 14, T = 8, AXH = 22, GAP = 10, SH = Math.round(Math.min(80, Math.max(34, H * 0.2)));
  const y0 = T, y1 = H - AXH - SH - GAP, sy0 = y1 + GAP, sy1 = sy0 + SH;
  const g = [];
  const now = R?.clock, x0 = R?.vit[0]?.t_sec ?? R?.from ?? 0, x1 = Math.max(x0 + 1800, (now ?? x0) + 120);
  const x = lin(x0, x1, ML, W - MR), yr = lin(0, 1, y1, y0), ym = lin(40, 120, sy1, sy0);
  g.push(s("defs", {}, s("clipPath", { id: "clip-r" }, s("rect", { x: ML, y: y0 - 2, width: W - ML - MR, height: y1 - y0 + 4 })),
    s("clipPath", { id: "clip-m" }, s("rect", { x: ML, y: sy0, width: W - ML - MR, height: SH }))));
  for (const v of [0, 0.2, 0.4, 0.6, 0.8, 1]) g.push(s("line", { class: "grid", x1: ML, x2: W - MR, y1: yr(v), y2: yr(v) }),
    s("text", { class: "ylab", x: ML - 6, y: yr(v) + 4, "text-anchor": "end" }, v.toFixed(1)));
  g.push(s("rect", { class: "strip-bg", x: ML, y: sy0, width: W - ML - MR, height: SH }),
    s("text", { class: "ylab", x: ML - 6, y: sy0 + 10, "text-anchor": "end" }, "MAP"));
  const span = x1 - x0, stepT = [300, 600, 900, 1800, 3600].find((d) => span / d <= (W < 600 ? 5 : 9)) ?? 7200;
  for (let t = Math.ceil(x0 / stepT) * stepT; t <= x1; t += stepT)
    g.push(s("line", { class: "grid", x1: x(t), x2: x(t), y1: y0, y2: sy1 }), s("text", { x: x(t), y: sy1 + 16, "text-anchor": x(t) > W - MR - 20 ? "end" : "middle" }, hm(t)));
  if (!R || now == null) {
    g.push(s("text", { class: "empty-label", x: (ML + W - MR) / 2, y: (y0 + y1) / 2, "text-anchor": "middle" },
      !R ? "No surgery selected" : playing() ? "Starting…" : "Press Play or the space bar to start"));
    return box.replaceChildren(s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" }, g));
  }
  for (const o of R.onsets) { // a real drop, shaded from its onset up to the clock, never ahead of it
    const e = Math.min(dropEnd(o), now);
    g.push(s("rect", { class: "drop", x: x(o.t_sec), y: y0, width: Math.max(2, x(e) - x(o.t_sec)), height: sy1 - y0 }),
      s("line", { class: "onset", x1: x(o.t_sec), x2: x(o.t_sec), y1: y0, y2: sy1 }));
  }
  const thr = R.meta?.threshold;
  if (thr != null) g.push(s("line", { class: "thr", x1: ML, x2: W - MR, y1: yr(thr), y2: yr(thr) }),
    s("text", { class: "thr-label", x: ML + 6, y: yr(thr) - 6 }, `Alarm threshold (${thr.toFixed(2)})`));
  g.push(s("line", { class: "map-thr", x1: ML, x2: W - MR, y1: ym(MAP_THR), y2: ym(MAP_THR) }),
    s("text", { class: "map-thr-label", x: W - MR - 4, y: ym(MAP_THR) - 3, "text-anchor": "end" }, "65 mmHg"),
    s("path", { class: "line s-map", "clip-path": "url(#clip-m)", d: pathD(R.vit.map((v) => [v.t_sec, v.map]), x, ym) }));
  const series = [["p_baseline1_10", "s-rule"], ["p_snapshot_10", "s-snap"], ["p10", "s-fus"]];
  for (const [key, cls] of series) {
    g.push(s("path", { class: `line ${cls}`, "clip-path": "url(#clip-r)", d: pathD(R.ticks.map((k) => [k.t_sec, k[key]]), x, yr) }));
    const k = lastTick();
    if (k?.[key] != null) g.push(s("circle", { class: `end ${cls}`, cx: x(k.t_sec), cy: yr(k[key]), r: key === "p10" ? 5 : 4 }));
  }
  for (const a of R.alarms) g.push(s("line", { class: "alarm-line", x1: x(a.t_sec), x2: x(a.t_sec), y1: y0, y2: y1 }),
    s("circle", { class: "alarm-dot", cx: x(a.t_sec), cy: yr(a.p10), r: 6 }));
  // alarm -> onset brackets ("warned N min earlier") and the empty 10 min before a missed drop: each label takes the first
  // row it fits in, a new row while the plot has room (top half), else the row that ends first
  const rows = [], maxRows = Math.max(1, Math.floor((y1 - y0 - 30) / 2 / 26) + 1);
  measure.font = `700 13px ${getComputedStyle(box).fontFamily}`;
  for (const o of R.onsets.filter((o) => o.warnable)) {
    const warned = o.warned && o.alarm_t_sec != null;
    const bx1 = x(Math.max(warned ? o.alarm_t_sec : o.t_sec - AHEAD_S, x0)), bx2 = x(o.t_sec);
    const text = warned ? `warned ${fmt(o.lead_min, 1)} min earlier` : "missed";
    const w = measure.measureText(text).width + 10, cx = Math.min(Math.max((bx1 + bx2) / 2, ML + w / 2), W - MR - w / 2);
    let row = rows.findIndex((r) => cx - w / 2 > r + 6);
    if (row < 0) row = rows.length < maxRows ? rows.length : rows.indexOf(Math.min(...rows));
    rows[row] = cx + w / 2;
    const by = y0 + 30 + row * 26;
    g.push(s("path", { class: warned ? "bracket" : "missed", d: `M${bx1.toFixed(1)},${by + 7}V${by}H${bx2.toFixed(1)}V${by + 7}` }),
      s("text", { class: warned ? "lead-label" : "missed-label", x: cx, y: by - 6, "text-anchor": "middle" }, text));
  }
  g.push(s("line", { class: "cursor", x1: x(now), x2: x(now), y1: y0, y2: sy1 }));
  const desc = `Surgery ${R.id} at ${hm(now)}: ${R.alarms.length} alarms and ${R.onsets.length} drops so far.`;
  box.replaceChildren(s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": desc }, g));
}

function scores() {
  const k = lastTick();
  const bar = (label, v, cls) => h("div", { class: `brow ${cls}` }, h("span", { class: "bl" }, label),
    h("span", { class: "bt" }, h("span", { class: "bf", style: `width:${v == null ? 0 : (v * 100).toFixed(1)}%` })), h("span", { class: "bv" }, pc(v)));
  $("scores-at").textContent = k ? (k.scored ? `10-min risk at ${hm(k.t_sec)}` : `not scored at ${hm(k.t_sec)}`) : "";
  $("bars").replaceChildren(
    h("div", { class: "bgroup" }, bar("Current-MAP rule", k?.p_baseline1_10, "rule"), bar("XGBoost snapshot", k?.p_snapshot_10, "snap"), bar("Fusion (10 min)", k?.p10, "fus")),
    h("div", { class: "bgroup" }, bar("Fusion 5 min", k?.p5, "h5"), bar("Fusion 10 min", k?.p10, "h10"), bar("Fusion 15 min", k?.p15, "h15")));
  const ons = R ? R.onsets.filter((o) => o.warnable) : [], warned = ons.filter((o) => o.warned);
  const med = median(warned.map((o) => o.lead_min)), nFalse = R ? R.alarms.filter((a) => settled(a) && !isTrue(a)).length : 0;
  const stat = (num, label) => h("div", { class: "stat" }, h("p", { class: "s-num" }, num), h("p", { class: "s-label" }, label));
  const e = R?.end;
  $("board").replaceChildren(
    stat(R?.clock != null ? `${warned.length}/${ons.length}` : "—", "drops warned / so far"),
    stat(R?.clock != null ? fmt(nFalse) : "—", "false alarms"),
    stat(med == null ? "—" : fmt(med, 1), "median warning, min"),
    h("p", { class: "b-cap" }, e ? `Whole surgery: ${e.warned} of ${e.events} drops warned, ${e.alarms} alarms, ${e.false_alarms} false — the same verdict as its locked test result.`
      : `${R?.from ? `Since ${hm(R.from)}. ` : ""}False alarm: no drop within 10 min of it.`));
  const x = R && info(R.id);
  $("caption").textContent = x?.reason || (x ? `Surgery ${x.caseid}: ${dur(x.duration_min)}, ${drops(x.n_events)}, one of the ${surgeries.length} unseen test surgeries.` : "");
  $("caption").title = $("caption").textContent; // the full text when a short window cuts it to one line
}

function bar() {
  const play = $("play");
  const icon = playing() ? "M6 4h4v16H6zm8 0h4v16h-4z" : "M7 4l13 8-13 8z";
  play.replaceChildren(s("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" }, s("path", { d: icon })));
  play.setAttribute("aria-label", playing() ? "Pause (space bar)" : "Play (space bar)");
  for (const id of ["play", "restart", "step"]) $(id).disabled = !R;
  const T = total(), now = R?.clock ?? R?.from ?? 0;
  $("clock").textContent = `${hm(now)} / ${hm(T)}`;
  $("fill").style.width = T ? `${Math.min(100, (100 * now) / T).toFixed(2)}%` : "0";
  const p = $("progress");
  p.setAttribute("aria-valuemin", "0");
  p.setAttribute("aria-valuemax", String(Math.round(T || 0)));
  p.setAttribute("aria-valuenow", String(now));
  p.setAttribute("aria-valuetext", hm(now));
  for (const b of document.querySelectorAll("[data-speed]")) b.setAttribute("aria-pressed", String(+b.dataset.speed === speed));
  const nd = nextDrop(), skip = $("skip");
  skip.disabled = nd == null;
  skip.title = R?.file === null ? "This surgery's drop times were not exported (no case file)" : nd == null ? "No later drop in this surgery" : `Presenter shortcut: jump to ${hm(nd)}`;
  const src = health?.data_source;
  $("status").textContent = R?.status || "";
  $("source").textContent = src ? `Vital signs read from ${src === "tiger" ? "Tiger Cloud (TimescaleDB)" : "local Parquet files"}` : "";
}

function wire() {
  $("play").onclick = toggle;
  $("step").onclick = step;
  $("restart").onclick = () => R && seek(0, true);
  $("skip").onclick = () => { const t = nextDrop(); if (t != null) seek(t, true); };
  const sp = document.querySelector(".speeds");
  for (const v of SPEEDS) {
    const b = h("button", { type: "button", "data-speed": v, "aria-pressed": "false" }, `${v}×`);
    b.onclick = () => setSpeed(v);
    sp.append(b);
  }
  const p = $("progress");
  p.onclick = (e) => {
    const T = total(), r = p.getBoundingClientRect();
    if (T) seek(Math.floor(((e.clientX - r.left) / r.width) * T / TICK_S) * TICK_S);
  };
  p.onkeydown = (e) => {
    const d = { ArrowRight: 300, ArrowLeft: -300 }[e.key];
    if (d && R) { e.preventDefault(); seek(Math.min(Math.max(0, (R.clock ?? R.from) + d), total() || 0)); }
  };
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || e.target.closest("select, input, textarea")) return;
    e.preventDefault();
    toggle();
  });
  addEventListener("resize", paint);
}

async function main() {
  wire();
  const [list, idx, hl] = await Promise.all([get(`${API}surgeries`), get(`${DATA}cases/index.json`), get(`${API}health`)]);
  surgeries = list?.surgeries || [];
  index = idx || [];
  health = hl;
  tabs();
  const want = +q.get("case"), id = info(want) ? want : surgeries[0]?.caseid;
  if (id != null) choose(id, q.get("autoplay") === "1", Math.floor(Math.max(0, +q.get("start") || 0) / TICK_S) * TICK_S);
  paint();
}
main();
