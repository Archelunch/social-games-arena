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
    const muX = Math.max(fillW, (r.mu / maxDomain) * 100);
    const bar = s("svg", {
      class: "skillbar",
      viewBox: "0 0 100 12",
      preserveAspectRatio: "none",
      role: "img",
      "aria-label": shortName(r.model) + " skill " + r.skill.toFixed(1),
    });
    bar.append(s("rect", { class: "skillbar__track", x: 0, y: 4, width: 100, height: 4, rx: 2 }));
    const fill = s("rect", { class: "skillbar__fill bar-anim", x: 0, y: 3, width: fillW, height: 6, rx: 2 });
    animDelay(fill, i);
    bar.append(fill);
    // uncertainty whisker: conservative skill -> optimistic mu
    bar.append(s("line", { class: "skillbar__whisker", x1: fillW, y1: 6, x2: muX, y2: 6 }));
    bar.append(s("line", { class: "skillbar__whisker", x1: muX, y1: 3, x2: muX, y2: 9 }));
    const skillCell = h("div", { class: "skillcell" }, bar, h("span", { class: "skillcell__val", text: fixed(r.skill, 1) }));
    const wolfRec = r.wolf_wins + dash + (r.wolf_games - r.wolf_wins);
    const villageRec = r.village_wins + dash + (r.village_games - r.village_wins);

    tbody.append(
      h("tr", null,
        h("td", { class: "lb__rank", text: String(i + 1) }),
        h("td", null, modelEl(r.model)),
        h("td", { class: "lb__skill" }, skillCell),
        h("td", { class: "num lb__wolf", title: r.wolf_wins + " wins of " + r.wolf_games + " as werewolves", text: wolfRec }),
        h("td", { class: "num lb__village", title: r.village_wins + " wins of " + r.village_games + " as villagers", text: villageRec }),
        h("td", { class: "num", text: int(r.games) }),
      ),
    );
  });
  const table = h("table", { class: "lb" },
    h("caption", { text: "Conservative skill, record split by the side played (wins–losses), and games rated." }),
    h("thead", null, h("tr", null,
      h("th", { scope: "col", text: "#" }),
      h("th", { scope: "col", text: "Model" }),
      h("th", { scope: "col", text: "Skill" }),
      h("th", { scope: "col", class: "num lb__wolf", text: "As wolf" }),
      h("th", { scope: "col", class: "num lb__village", text: "As village" }),
      h("th", { scope: "col", class: "num", text: "Games" }),
    )),
    tbody,
  );
  mount("leaderboard", table);
}

function renderScatter(split) {
  const pts = split.models.filter((m) => m.exile_accuracy != null);
  if (!pts.length) {
    mount("scatter", h("p", { class: "panel__note", text: "Not enough resolved exiles to plot yet." }));
    return;
  }

  const ML = 12, MR = 8, MT = 8, MB = 12, W = 100, H = 100;
  const pw = W - ML - MR, ph = H - MT - MB;
  const px = (v) => ML + v * pw;
  const py = (v) => MT + (1 - v) * ph;

  const svg = s("svg", { class: "scatter", viewBox: "0 0 100 100", role: "img",
    "aria-label": "Wolf win rate against exile accuracy, one numbered point per model; identities in the legend." });

  // plot frame + 0.5 quadrant dividers + balance diagonal
  svg.append(s("rect", { class: "sc-axis", x: ML, y: MT, width: pw, height: ph, fill: "none" }));
  svg.append(s("line", { class: "sc-mid", x1: px(0.5), y1: MT, x2: px(0.5), y2: MT + ph }));
  svg.append(s("line", { class: "sc-mid", x1: ML, y1: py(0.5), x2: ML + pw, y2: py(0.5) }));
  svg.append(s("line", { class: "sc-diag", x1: px(0), y1: py(0), x2: px(1), y2: py(1) }));

  // population mean crosshair (field average)
  if (split.exile_accuracy != null) {
    svg.append(s("line", { class: "sc-mean", x1: px(split.exile_accuracy), y1: MT, x2: px(split.exile_accuracy), y2: MT + ph }));
  }
  svg.append(s("line", { class: "sc-mean", x1: ML, y1: py(split.wolf_win_rate), x2: ML + pw, y2: py(split.wolf_win_rate) }));

  // ticks
  for (const t of [0, 0.5, 1]) {
    svg.append(s("text", { class: "sc-tick", x: px(t), y: MT + ph + 3.4, "text-anchor": "middle", text: t }));
    svg.append(s("text", { class: "sc-tick", x: ML - 1.6, y: py(t) + 0.9, "text-anchor": "end", text: t }));
  }
  // axis labels
  svg.append(s("text", { class: "sc-axislabel", x: ML + pw / 2, y: H - 0.5, "text-anchor": "middle", text: "detection: exile accuracy →" }));
  const yl = s("text", { class: "sc-axislabel", x: 3, y: MT + ph / 2, "text-anchor": "middle", text: "deception: wolf win rate →" });
  yl.setAttribute("transform", "rotate(-90 3 " + (MT + ph / 2) + ")");
  svg.append(yl);
  // quadrant hints
  svg.append(s("text", { class: "sc-quad", x: ML + 1.5, y: MT + 3, text: "deceives, misses" }));
  svg.append(s("text", { class: "sc-quad", x: ML + pw - 1.5, y: MT + 3, "text-anchor": "end", text: "all-rounder" }));
  svg.append(s("text", { class: "sc-quad", x: ML + 1.5, y: MT + ph - 1.5, text: "weak both" }));
  svg.append(s("text", { class: "sc-quad", x: ML + pw - 1.5, y: MT + ph - 1.5, "text-anchor": "end", text: "detects only" }));

  // De-clump: nudge coincident points apart so every numbered marker stays legible.
  const nodes = pts.map((m) => ({ m, tx: px(m.exile_accuracy), ty: py(m.wolf_win_rate), x: px(m.exile_accuracy), y: py(m.wolf_win_rate) }));
  const MIND = 6.2;
  for (let it = 0; it < 80; it++) {
    for (let a = 0; a < nodes.length; a++) {
      for (let b = a + 1; b < nodes.length; b++) {
        let dx = nodes[b].x - nodes[a].x, dy = nodes[b].y - nodes[a].y, d = Math.hypot(dx, dy);
        if (d === 0) { dx = 1; dy = 1; d = Math.SQRT2; }
        if (d < MIND) {
          const k = (MIND - d) / 2 / d;
          nodes[a].x -= dx * k; nodes[a].y -= dy * k;
          nodes[b].x += dx * k; nodes[b].y += dy * k;
        }
      }
    }
    for (const n of nodes) {
      n.x = Math.max(ML + 3.2, Math.min(ML + pw - 3.2, n.x));
      n.y = Math.max(MT + 3.2, Math.min(MT + ph - 3.2, n.y));
    }
  }
  // leader line from the true position to the nudged marker (honest about the offset)
  for (const n of nodes) {
    if (Math.hypot(n.x - n.tx, n.y - n.ty) > 0.8) {
      svg.append(s("line", { class: "sc-leader", x1: n.tx, y1: n.ty, x2: n.x, y2: n.y }));
      svg.append(s("circle", { class: "sc-anchor", cx: n.tx, cy: n.ty, r: 0.7 }));
    }
  }
  nodes.forEach((n, i) => {
    const g = s("g", { class: "sc-node sc-dot-anim", role: "img",
      "aria-label": shortName(n.m.model) + ": deception " + pct(n.m.wolf_win_rate) + ", detection " + pct(n.m.exile_accuracy) });
    animDelay(g, i);
    g.append(s("circle", { class: "sc-point", cx: n.x, cy: n.y, r: 3.1 }));
    g.append(s("text", { class: "sc-num", x: n.x, y: n.y + 1.15, "text-anchor": "middle", text: String(i + 1) }));
    svg.append(g);
  });

  const legend = h("ol", { class: "sc-legend" });
  nodes.forEach((n, i) => {
    legend.append(h("li", { class: "sc-legend__item" },
      h("span", { class: "sc-legend__num", text: String(i + 1) }),
      h("span", { class: "sc-legend__name" }, modelEl(n.m.model)),
      h("span", { class: "sc-legend__stat" },
        "deception ", h("b", { text: pct(n.m.wolf_win_rate) }),
        " · detection ", h("b", { text: pct(n.m.exile_accuracy) })),
    ));
  });

  mount("scatter", h("div", { class: "scatter-wrap" }, svg, legend, scatterFallback(pts)));
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
    statBox("Avg game length", fixed(cost.mean_game_length, 1) + " rounds"),
    statBox("Illegal-move rate", pct(cost.mean_illegal_move_rate)),
    statBox("Spend", t.cost_usd == null ? "—" : "$" + fixed(t.cost_usd, 2)),
  );

  const tbody = h("tbody");
  for (const m of cost.models) {
    const tokensPerGame = m.games ? (m.prompt + m.completion) / m.games : 0;
    const illegalRate = m.tool_calls ? m.illegal_moves / m.tool_calls : 0;
    tbody.append(h("tr", null,
      h("td", null, modelEl(m.model)),
      h("td", { class: "num", text: pct(m.win_rate) }),
      h("td", { class: "num", text: int(tokensPerGame) }),
      h("td", { class: "num" + (illegalRate > 0 ? " illegal-hot" : ""), text: pct(illegalRate) }),
    ));
  }
  const table = h("table", { class: "cost" },
    h("caption", { text: "Per-model spend and rule discipline. Win rate is across all games." }),
    h("thead", null, h("tr", null,
      h("th", { scope: "col", text: "Model" }),
      h("th", { scope: "col", class: "num", text: "Win rate" }),
      h("th", { scope: "col", class: "num", text: "Tokens / game" }),
      h("th", { scope: "col", class: "num", text: "Illegal" }),
    )),
    tbody);
  mount("cost", h("div", null, totals, table));
}

function statBox(label, value) {
  return h("div", { class: "stat" }, h("dt", { text: label }), h("dd", { text: value }));
}

function renderSelfPlay(rows) {
  if (!rows.length) {
    mount("selfplay", h("p", { class: "panel__note", text: "No self-play games yet." }));
    return;
  }
  const grid = h("div", { class: "selfplay" });
  rows.forEach((r, i) => {
    grid.append(h("div", { class: "sp-card" },
      h("div", { class: "sp-card__head" },
        modelEl(r.model),
        h("span", { class: "sp-card__rounds", text: fixed(r.mean_rounds, 1) + " avg rounds" }),
      ),
      h("div", { class: "sp-metrics" },
        spMetric("Deception", "wolf", r.wolf_win_rate, i),
        spMetric("Detection", "village", r.exile_accuracy, i),
      ),
    ));
  });
  mount("selfplay", grid);
}

function spMetric(label, pole, value, i) {
  const track = h("div", { class: "sp-track" });
  if (value != null) {
    const fill = h("div", { class: "sp-fill sp-fill--" + pole + " bar-anim", style: "width:" + value * 100 + "%" });
    animDelay(fill, i);
    track.append(fill);
  }
  return h("div", { class: "sp-metric" },
    h("span", { class: "sp-metric__label", text: label }),
    track,
    h("span", { class: "sp-metric__val", text: pct(value) }));
}

function renderFooter(meta) {
  const node = document.getElementById("footer-meta");
  node.textContent =
    "models: " + meta.models.map(shortName).join(", ") +
    "  ·  runs: " + meta.run_dirs.join(", ") +
    "  ·  " + meta.n_games + " games, " + meta.n_rated + " rated, " + meta.n_skipped + " self-play";
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
