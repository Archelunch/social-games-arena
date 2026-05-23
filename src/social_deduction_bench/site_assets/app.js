"use strict";

// The Werewolf Benchmark — static results renderer.
// Fetches the pre-aggregated data.json and draws each panel with hand-rolled inline
// SVG. No framework, no network. Model names contain "/", so every name reaches the
// DOM via textContent / data- attributes, never as an id or selector.

const SVG_NS = "http://" + "www.w3.org/2000/svg";
const REDUCE_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// --- tiny DOM helpers ----------------------------------------------------

function h(tag, attrs, ...kids) {
  const e = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "class") e.className = v;
      else if (k === "text") e.textContent = v;
      else e.setAttribute(k, String(v));
    }
  }
  append(e, kids);
  return e;
}

function s(tag, attrs, ...kids) {
  const e = document.createElementNS(SVG_NS, tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null) continue;
      if (k === "text") e.textContent = String(v);
      else e.setAttribute(k, String(v));
    }
  }
  append(e, kids);
  return e;
}

function append(parent, kids) {
  for (const kid of kids) {
    if (kid == null || kid === false) continue;
    parent.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
}

function mount(id, node) {
  const host = document.getElementById(id);
  host.replaceChildren(node);
}

// --- formatting ----------------------------------------------------------

const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");
const fixed = (x, d) => (x == null ? "—" : Number(x).toFixed(d));
const int = (x) => (x == null ? "—" : Math.round(x).toLocaleString("en-US"));
const dash = "–"; // en dash for records

function modelEl(name) {
  const slash = name.lastIndexOf("/");
  const span = h("span", { class: "model", "data-model": name });
  if (slash >= 0) {
    span.append(h("span", { class: "model__provider", text: name.slice(0, slash + 1) }));
    span.append(h("span", { class: "model__name", text: name.slice(slash + 1) }));
  } else {
    span.append(h("span", { class: "model__name", text: name }));
  }
  return span;
}

function shortName(name) {
  const slash = name.lastIndexOf("/");
  return slash >= 0 ? name.slice(slash + 1) : name;
}

function animDelay(node, i) {
  if (!REDUCE_MOTION) node.style.animationDelay = i * 45 + "ms";
}

// --- panels --------------------------------------------------------------

function renderDateline(meta) {
  const node = document.getElementById("dateline");
  node.replaceChildren(
    h("b", { text: int(meta.n_games) }), " games · ",
    h("b", { text: int(meta.models.length) }), " models · ",
    h("b", { text: int(meta.n_rated) }), " rated · ",
    h("b", { text: int(meta.n_skipped) }), " self-play",
  );
}

function renderLeaderboard(board) {
  if (!board.length) {
    mount("leaderboard", h("p", { class: "panel__note", text: "No rated games yet." }));
    return;
  }
  const maxDomain = Math.max(...board.map((r) => r.mu)) * 1.05 || 1;
  const tbody = h("tbody");
  board.forEach((r, i) => {
    const fillW = Math.max(0, (r.skill / maxDomain) * 100);
    const muX = Math.max(0, (r.mu / maxDomain) * 100);
    const bar = s("svg", { class: "skillbar", viewBox: "0 0 100 16", role: "img",
      "aria-label": shortName(r.model) + " skill " + r.skill.toFixed(1) });
    bar.append(s("rect", { class: "skillbar__track", x: 0, y: 6, width: 100, height: 4, rx: 2 }));
    const fill = s("rect", { class: "skillbar__fill bar-anim", x: 0, y: 5, width: fillW, height: 6, rx: 2 });
    animDelay(fill, i);
    bar.append(fill);
    // uncertainty whisker: conservative skill -> mu
    bar.append(s("line", { class: "skillbar__whisker", x1: fillW, y1: 8, x2: muX, y2: 8 }));
    bar.append(s("line", { class: "skillbar__whisker", x1: muX, y1: 5, x2: muX, y2: 11 }));

    tbody.append(
      h("tr", null,
        h("td", { class: "lb__rank", text: String(i + 1) }),
        h("td", null, modelEl(r.model)),
        h("td", { class: "lb__bar" }, bar),
        h("td", { class: "num", text: fixed(r.skill, 1) }),
        h("td", { class: "num", text: r.wins + dash + r.losses }),
        h("td", { class: "num", text: int(r.games) }),
      ),
    );
  });
  const table = h("table", { class: "lb" },
    h("caption", { text: "Conservative skill, win–loss, and games rated per model." }),
    h("thead", null, h("tr", null,
      h("th", { scope: "col", text: "#" }),
      h("th", { scope: "col", text: "Model" }),
      h("th", { scope: "col", text: "Skill" }),
      h("th", { scope: "col", class: "num", text: "μ−3σ" }),
      h("th", { scope: "col", class: "num", text: "W–L" }),
      h("th", { scope: "col", class: "num", text: "Games" }),
    )),
    tbody,
  );
  mount("leaderboard", table);
}

function renderScatter(split) {
  const pts = split.models.filter((m) => m.exile_accuracy != null);
  const wrap = h("div", { class: "scatter-wrap" });
  if (!pts.length) {
    mount("scatter", h("p", { class: "panel__note", text: "Not enough resolved exiles to plot yet." }));
    return;
  }

  const ML = 13, MR = 6, MT = 6, MB = 13, W = 100, H = 100;
  const pw = W - ML - MR, ph = H - MT - MB;
  const px = (v) => ML + v * pw;
  const py = (v) => MT + (1 - v) * ph;

  const svg = s("svg", { class: "scatter", viewBox: "0 0 100 100", role: "img",
    "aria-label": "Scatter of wolf win rate against exile accuracy, one point per model." });

  // plot frame + 0.5 quadrant dividers + balance diagonal
  svg.append(s("rect", { class: "sc-axis", x: ML, y: MT, width: pw, height: ph, fill: "none" }));
  svg.append(s("line", { class: "sc-mid", x1: px(0.5), y1: MT, x2: px(0.5), y2: MT + ph }));
  svg.append(s("line", { class: "sc-mid", x1: ML, y1: py(0.5), x2: ML + pw, y2: py(0.5) }));
  svg.append(s("line", { class: "sc-diag", x1: px(0), y1: py(0), x2: px(1), y2: py(1) }));

  // population mean crosshair
  if (split.exile_accuracy != null) {
    svg.append(s("line", { class: "sc-mean", x1: px(split.exile_accuracy), y1: MT, x2: px(split.exile_accuracy), y2: MT + ph }));
  }
  svg.append(s("line", { class: "sc-mean", x1: ML, y1: py(split.wolf_win_rate), x2: ML + pw, y2: py(split.wolf_win_rate) }));

  // ticks
  for (const t of [0, 0.5, 1]) {
    svg.append(s("text", { class: "sc-tick", x: px(t), y: MT + ph + 3.2, "text-anchor": "middle", text: t }));
    svg.append(s("text", { class: "sc-tick", x: ML - 1.5, y: py(t) + 0.9, "text-anchor": "end", text: t }));
  }
  // axis labels
  svg.append(s("text", { class: "sc-axislabel", x: ML + pw / 2, y: H - 1, "text-anchor": "middle", text: "exile accuracy (detection) →" }));
  const yl = s("text", { class: "sc-axislabel", x: 3.5, y: MT + ph / 2, "text-anchor": "middle", text: "wolf win rate (deception) →" });
  yl.setAttribute("transform", "rotate(-90 3.5 " + (MT + ph / 2) + ")");
  svg.append(yl);
  // quadrant hints
  svg.append(s("text", { class: "sc-quad", x: ML + 1, y: MT + 3, text: "deceives, misses wolves" }));
  svg.append(s("text", { class: "sc-quad", x: ML + pw - 1, y: MT + 3, "text-anchor": "end", text: "all-rounder" }));
  svg.append(s("text", { class: "sc-quad", x: ML + 1, y: MT + ph - 1, text: "weak both" }));
  svg.append(s("text", { class: "sc-quad", x: ML + pw - 1, y: MT + ph - 1, "text-anchor": "end", text: "detects, can't deceive" }));

  pts.forEach((m, i) => {
    const cx = px(m.exile_accuracy), cy = py(m.wolf_win_rate);
    const dot = s("circle", { class: "sc-point sc-dot-anim", cx, cy, r: 1.7 });
    animDelay(dot, i);
    const hit = s("circle", { class: "sc-hit", cx, cy, r: 4, tabindex: "0", role: "img",
      "aria-label": shortName(m.model) + ": wolf win " + pct(m.wolf_win_rate) + ", exile accuracy " + pct(m.exile_accuracy) });
    const tip = () =>
      shortName(m.model) + " · deception " + pct(m.wolf_win_rate) + " · detection " + pct(m.exile_accuracy);
    hit.addEventListener("mouseenter", (e) => showTip(tip(), e.clientX, e.clientY));
    hit.addEventListener("mousemove", (e) => showTip(tip(), e.clientX, e.clientY));
    hit.addEventListener("mouseleave", hideTip);
    hit.addEventListener("focus", () => {
      const r = hit.getBoundingClientRect();
      showTip(tip(), r.left + r.width / 2, r.top);
    });
    hit.addEventListener("blur", hideTip);
    svg.append(dot, hit);
    svg.append(s("text", { class: "sc-label", x: cx + 2.4, y: cy - 1.6, text: shortName(m.model) }));
  });

  wrap.append(svg, scatterFallback(pts));
  mount("scatter", wrap);
}

function scatterFallback(pts) {
  const tbody = h("tbody");
  for (const m of pts) {
    tbody.append(h("tr", null,
      h("th", { scope: "row", text: shortName(m.model) }),
      h("td", { text: pct(m.wolf_win_rate) }),
      h("td", { text: pct(m.exile_accuracy) }),
    ));
  }
  // Wrap in a div: a width:1px <table> sizes to content and (being absolute) would
  // escape the scroll container and widen the page; a div clips it to 1px.
  return h("div", { class: "visually-hidden" },
    h("table", null,
      h("caption", { text: "Deceiver vs detector, tabular." }),
      h("thead", null, h("tr", null,
        h("th", { scope: "col", text: "Model" }),
        h("th", { scope: "col", text: "Wolf win rate" }),
        h("th", { scope: "col", text: "Exile accuracy" }),
      )),
      tbody));
}

function renderHeadToHead(h2h) {
  const models = h2h.models;
  if (!models.length) {
    mount("headtohead", h("p", { class: "panel__note", text: "Only self-play games so far — no cross-model matchups to compare." }));
    return;
  }
  const map = new Map();
  for (const c of h2h.cells) map.set(c.row + "|" + c.col, c);

  const headRow = h("tr", null, h("td", { "aria-hidden": "true" }));
  for (const c of models) headRow.append(h("th", { scope: "col", text: shortName(c) }));
  const tbody = h("tbody");
  for (const r of models) {
    const tr = h("tr", null, h("th", { scope: "row", text: shortName(r) }));
    for (const c of models) {
      if (r === c) {
        tr.append(h("td", { class: "h2h__diag" }, h("a", { href: "#h-selfplay", title: "self-play", text: "◐" })));
        continue;
      }
      let rWins = 0, cWins = 0, games = 0;
      const fwd = map.get(r + "|" + c);
      const rev = map.get(c + "|" + r);
      if (fwd) { rWins = fwd.row_wins; cWins = fwd.col_wins; games = fwd.games; }
      else if (rev) { rWins = rev.col_wins; cWins = rev.row_wins; games = rev.games; }
      if (!games) { tr.append(h("td", { class: "h2h__none", text: "·" })); continue; }
      const cls = rWins > cWins ? "h2h__row-leads" : rWins < cWins ? "h2h__col-leads" : "h2h__even";
      tr.append(h("td", { class: cls, text: rWins + dash + cWins }));
    }
    tbody.append(tr);
  }
  const table = h("table", { class: "h2h" },
    h("caption", { text: "Row model's wins – column model's wins, head to head." }),
    h("thead", null, headRow),
    tbody);
  mount("headtohead", table);
}

function renderCost(cost) {
  const t = cost.totals;
  const totalTokens = (t.prompt || 0) + (t.completion || 0);
  const totals = h("dl", { class: "cost-totals" },
    statBox("Total tokens", int(totalTokens)),
    statBox("Tool calls", int(t.tool_calls)),
    statBox("Illegal-move rate", pct(cost.mean_illegal_move_rate)),
    statBox("Avg game length", fixed(cost.mean_game_length, 1)),
    statBox("Spend", t.cost_usd == null ? "—" : "$" + fixed(t.cost_usd, 2)),
  );

  const tbody = h("tbody");
  for (const m of cost.models) {
    const tokensPerGame = m.games ? (m.prompt + m.completion) / m.games : 0;
    const illegalRate = m.tool_calls ? m.illegal_moves / m.tool_calls : 0;
    const promptShare = (m.prompt + m.completion) ? (m.prompt / (m.prompt + m.completion)) * 100 : 50;
    const split = h("span", { class: "split-bar", title: "prompt vs completion tokens", "aria-hidden": "true" },
      h("i", { class: "prompt", style: "width:" + promptShare + "%" }),
      h("i", { class: "completion", style: "width:" + (100 - promptShare) + "%" }));
    tbody.append(h("tr", null,
      h("td", null, modelEl(m.model)),
      h("td", { class: "num", text: int(m.games) }),
      h("td", { class: "num", text: pct(m.win_rate) }),
      h("td", { class: "num", text: int(tokensPerGame) }),
      h("td", { class: "num" + (illegalRate > 0 ? " illegal-hot" : ""), text: pct(illegalRate) }),
      h("td", null, split),
    ));
  }
  const table = h("table", { class: "cost" },
    h("thead", null, h("tr", null,
      h("th", { scope: "col", text: "Model" }),
      h("th", { scope: "col", class: "num", text: "Games" }),
      h("th", { scope: "col", class: "num", text: "Win rate" }),
      h("th", { scope: "col", class: "num", text: "Tokens / game" }),
      h("th", { scope: "col", class: "num", text: "Illegal rate" }),
      h("th", { scope: "col", text: "Prompt : completion" }),
    )),
    tbody);
  mount("cost", h("div", null, totals, table));
}

function statBox(label, value) {
  return h("div", null, h("dt", { text: label }), h("dd", { text: value }));
}

function renderSelfPlay(rows) {
  if (!rows.length) {
    mount("selfplay", h("p", { class: "panel__note", text: "No self-play games yet." }));
    return;
  }
  const grid = h("div", { class: "selfplay" });
  rows.forEach((r, i) => {
    grid.append(h("div", { class: "sp-row" },
      modelEl(r.model),
      h("div", { class: "sp-bars" },
        spBar("deception", r.wolf_win_rate, "wolf", i),
        spBar("detection", r.exile_accuracy, "village", i),
      ),
      h("div", { class: "sp-rounds" }, fixed(r.mean_rounds, 1), h("span", { text: "avg rounds" })),
    ));
  });
  mount("selfplay", grid);
}

function spBar(label, value, pole, i) {
  const track = h("div", { class: "sp-track" });
  if (value != null) {
    const fill = h("div", { class: "sp-fill sp-fill--" + pole + " bar-anim", style: "width:" + value * 100 + "%" });
    animDelay(fill, i);
    track.append(fill);
  }
  return h("div", { class: "sp-bar" },
    h("span", { class: "label", text: label }),
    track,
    h("span", { class: "num", text: pct(value) }));
}

function renderFooter(meta) {
  const node = document.getElementById("footer-meta");
  node.textContent =
    "models: " + meta.models.map(shortName).join(", ") +
    "  ·  runs: " + meta.run_dirs.join(", ") +
    "  ·  " + meta.n_games + " games, " + meta.n_rated + " rated, " + meta.n_skipped + " self-play";
}

// --- tooltip -------------------------------------------------------------

let tipEl = null;
function showTip(text, x, y) {
  if (!tipEl) tipEl = document.getElementById("tooltip");
  tipEl.textContent = text;
  tipEl.hidden = false;
  const pad = 12;
  const w = tipEl.offsetWidth;
  let left = x + pad;
  if (left + w > window.innerWidth) left = x - w - pad;
  tipEl.style.left = Math.max(pad, left) + "px";
  tipEl.style.top = Math.max(pad, y + pad) + "px";
}
function hideTip() {
  if (tipEl) tipEl.hidden = true;
}

// --- boot ----------------------------------------------------------------

function showError(message) {
  const banner = document.getElementById("error");
  banner.textContent = message;
  banner.hidden = false;
}

function render(data) {
  renderDateline(data.meta);
  renderLeaderboard(data.leaderboard);
  renderScatter(data.deceiver_detector);
  renderHeadToHead(data.head_to_head);
  renderCost(data.cost_efficiency);
  renderSelfPlay(data.self_play);
  renderFooter(data.meta);
}

fetch('data.json')
  .then((r) => {
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  })
  .then(render)
  .catch((err) => {
    showError(
      "Could not load data.json (" + err.message + "). Serve this folder over a local " +
      "server, e.g. run  python -m http.server  from here, then open the printed address.",
    );
  });
