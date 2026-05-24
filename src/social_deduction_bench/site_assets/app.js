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
    h("a", { class: "dateline__link", href: "#h-games" }, h("b", { text: int(meta.n_games) })), " games · ",
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
  // Shared scale anchored to the data range (not 0) so rank gaps are visible and you
  // can see which models' ratings overlap. Dot = conservative skill; line reaches mu.
  const lo = Math.min(...board.map((r) => r.skill));
  const hi = Math.max(...board.map((r) => r.mu));
  const pad = (hi - lo) * 0.12 || 1;
  const d0 = lo - pad, d1 = hi + pad;
  const X = (v) => ((v - d0) / (d1 - d0)) * 100;
  const tbody = h("tbody");
  board.forEach((r, i) => {
    const sx = X(r.skill), mx = X(r.mu);
    const plot = h("div", { class: "skillplot" + (i === 0 ? " skillplot--top" : ""), "aria-hidden": "true" },
      h("span", { class: "skillplot__track" }),
      h("span", { class: "skillplot__range", style: "left:" + sx + "%;width:" + Math.max(0, mx - sx) + "%" }),
      h("span", { class: "skillplot__dot", style: "left:" + sx + "%" }),
    );
    const skillCell = h("div", { class: "skillcell" }, plot, h("span", { class: "skillcell__val", text: fixed(r.skill, 1) }));
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
    h("caption", { class: "visually-hidden", text: "Model skill, record split by side, and games rated." }),
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
    mount("scatter", h("p", { class: "panel__note", text: "No resolved exiles yet, so there is nothing to plot." }));
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
  svg.append(s("text", { class: "sc-quad", x: ML + 1.5, y: MT + 3, text: "good liar" }));
  svg.append(s("text", { class: "sc-quad", x: ML + pw - 1.5, y: MT + 3, "text-anchor": "end", text: "all-rounder" }));
  svg.append(s("text", { class: "sc-quad", x: ML + 1.5, y: MT + ph - 1.5, text: "weak at both" }));
  svg.append(s("text", { class: "sc-quad", x: ML + pw - 1.5, y: MT + ph - 1.5, "text-anchor": "end", text: "good detective" }));

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
      h("caption", { text: "Each model's deception and detection rate." }),
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
    mount("headtohead", h("p", { class: "panel__note", text: "Only self-play games so far, so there are no matchups to compare yet." }));
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
    h("caption", { class: "visually-hidden", text: "Each model's record against every other, row versus column." }),
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
    statBox("Spend", t.cost_usd == null ? "—" : "$" + fixed(t.cost_usd, 2)),
  );

  const tbody = h("tbody");
  for (const m of cost.models) {
    const tokensPerGame = m.games ? (m.prompt + m.completion) / m.games : 0;
    tbody.append(h("tr", null,
      h("td", null, modelEl(m.model)),
      h("td", { class: "num", text: int(tokensPerGame) }),
      h("td", { class: "num", text: int(m.prompt + m.completion) }),
    ));
  }
  const table = h("table", { class: "cost" },
    h("caption", { class: "visually-hidden", text: "Token spend per model, per game and in total." }),
    h("thead", null, h("tr", null,
      h("th", { scope: "col", text: "Model" }),
      h("th", { scope: "col", class: "num", text: "Tokens / game" }),
      h("th", { scope: "col", class: "num", text: "Total tokens" }),
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

// --- replays: the game library ------------------------------------------

function factionOf(role) {
  return role === "werewolf" ? "wolf" : "village";
}

function side(model, pole) {
  return h("span", { class: "side side--" + pole }, h("span", { class: "side__pip", "aria-hidden": "true" }), modelEl(model));
}

// The masthead "Watch a game" shortcut opens the most dramatic match: the longest
// cross-model (non-self-play) game, tie-broken by lowest game_id so the pick is stable
// across reloads. Falls back to the longest game overall, then to null (no games).
function featuredGameId(games) {
  if (!games || !games.length) return null;
  const cross = games.filter((g) => g.wolf_model && g.village_model && g.wolf_model !== g.village_model);
  const pool = cross.length ? cross : games;
  let best = null;
  for (const g of pool) {
    if (!best || g.rounds > best.rounds || (g.rounds === best.rounds && g.game_id < best.game_id)) best = g;
  }
  return best ? best.game_id : null;
}

// Builds the filterable game list. Shared by the dashboard's Replays panel and the
// in-replay "Browse games" panel; `currentId` marks the game being watched (if any).
function gamesBrowser(games, currentId) {
  const models = [...new Set(games.flatMap((g) => [g.wolf_model, g.village_model]).filter((m) => m && m !== "mixed"))].sort();
  const modelSel = h("select", { class: "games__select", "aria-label": "Filter by model" },
    h("option", { value: "", text: "All models" }),
    ...models.map((m) => h("option", { value: m, text: shortName(m) })));
  const sideSel = h("select", { class: "games__select", "aria-label": "Filter by winning side" },
    h("option", { value: "", text: "Any outcome" }),
    h("option", { value: "werewolves", text: "Wolves win" }),
    h("option", { value: "villagers", text: "Village win" }));
  const count = h("span", { class: "games__count" });
  const list = h("div", { class: "games__list" });

  const draw = () => {
    const model = modelSel.value;
    const outcome = sideSel.value;
    list.replaceChildren();
    let shown = 0;
    for (const gm of games) {
      const wolf = gm.wolf_model || "mixed";
      const vil = gm.village_model || "mixed";
      if (model && wolf !== model && vil !== model) continue;
      if (outcome && gm.winner !== outcome) continue;
      shown += 1;
      const wolfWon = gm.winner === "werewolves";
      const current = gm.game_id === currentId;
      list.append(
        h("a", { class: "game-row" + (current ? " game-row--current" : ""),
          href: "#/game/" + encodeURIComponent(gm.game_id), "aria-current": current ? "true" : null },
          h("span", { class: "game-row__matchup" },
            side(wolf, "wolf"), h("span", { class: "game-row__vs", text: "vs" }), side(vil, "village")),
          h("span", { class: "badge badge--" + (wolfWon ? "wolf" : "village"), text: wolfWon ? "Wolves win" : "Village wins" }),
          h("span", { class: "game-row__rounds", text: gm.rounds + (gm.rounds === 1 ? " round" : " rounds") }),
          current
            ? h("span", { class: "game-row__watch game-row__watch--current" },
                h("span", { class: "game-row__watchlbl", text: "Watching" }))
            : h("span", { class: "game-row__watch" },
                h("span", { class: "game-row__play", "aria-hidden": "true", text: "▶" }),
                h("span", { class: "game-row__watchlbl", text: "Watch" })),
        ),
      );
    }
    count.textContent = shown + (shown === 1 ? " game" : " games");
    if (!shown) list.append(h("p", { class: "panel__note", text: "No games match these filters." }));
  };
  draw();
  modelSel.addEventListener("change", draw);
  sideSel.addEventListener("change", draw);

  return h("div", { class: "games" },
    h("div", { class: "games__filters" },
      h("label", { class: "games__field" }, h("span", { class: "games__label", text: "Model" }), modelSel),
      h("label", { class: "games__field" }, h("span", { class: "games__label", text: "Outcome" }), sideSel),
      count),
    list);
}

function renderGames(games) {
  if (!games || !games.length) {
    mount("games", h("p", { class: "panel__note", text: "No games to replay yet." }));
    return;
  }
  mount("games", gamesBrowser(games, null));
}

// --- replay: the game screen --------------------------------------------

const PHASE_GLYPH = { night: "☾", day: "☀" }; // waning moon, sun
const ARROW_KINDS = ["seer", "doctor", "accuse", "defend", "kill", "vote"];

function seatXY(i, n) {
  const a = -Math.PI / 2 + (i * 2 * Math.PI) / n;
  return { x: 50 + 39 * Math.cos(a), y: 50 + 41 * Math.sin(a) };
}

function deadThrough(events, idx) {
  const dead = new Set();
  for (let i = 0; i <= idx; i++) {
    const e = events[i];
    if (e.type === "kill_resolved" && e.payload.victim) dead.add(e.payload.victim);
    if (e.type === "exile_resolved" && e.payload.exiled) dead.add(e.payload.exiled);
  }
  return dead;
}

function ballotArrows(ballots, kind) {
  const out = [];
  for (const [voter, target] of Object.entries(ballots || {})) {
    if (!target || target === "abstain" || target === voter) continue;
    out.push({ from: voter, to: target, kind });
  }
  return out;
}

function arrowsFor(e, players, deadBefore) {
  const p = e.payload;
  if (e.type === "seer_inspect") return [{ from: e.actor, to: p.target, kind: "seer" }];
  if (e.type === "doctor_protect") return [{ from: e.actor, to: p.target, kind: "doctor" }];
  if (e.type === "accusation") return [{ from: e.actor, to: p.target, kind: "accuse" }];
  if (e.type === "defense") return [{ from: e.actor, to: p.defended, kind: "defend" }];
  if (e.type === "kill_ballots") return ballotArrows(p.ballots, "kill");
  if (e.type === "exile_resolved") return ballotArrows(p.ballots, "vote");
  if (e.type === "kill_resolved" && p.victim) {
    return players
      .filter((pl) => pl.role === "werewolf" && !deadBefore.has(pl.name) && pl.name !== p.victim)
      .map((pl) => ({ from: pl.name, to: p.victim, kind: "kill" }));
  }
  return [];
}

function mountReplay(g) {
  const root = document.getElementById("replay");
  const events = g.events;
  const players = g.players;
  const idxByName = {};
  const roleByName = {};
  players.forEach((p, i) => { idxByName[p.name] = i; roleByName[p.name] = p.role; });
  const pos = players.map((_, i) => seatXY(i, players.length));
  const st = { cursor: 0, playing: false, sel: null, timer: null, speed: 1 };

  // ---- top bar ----
  const phase = h("span", { class: "rp-phase" });
  const winnerWolf = g.winner === "werewolves";
  const topbar = h("div", { class: "rp-topbar" },
    h("a", { class: "rp-back", href: "#/", text: "← Replays" }),
    h("div", { class: "rp-matchup" }, side(g.wolf_model || "mixed", "wolf"),
      h("span", { class: "rp-matchup__vs", text: "vs" }), side(g.village_model || "mixed", "village")),
    h("div", { class: "rp-topbar__right" }, phase,
      h("span", { class: "badge badge--" + (winnerWolf ? "wolf" : "village"), text: winnerWolf ? "Wolves win" : "Village win" })),
  );

  // ---- round table ----
  const arrows = s("svg", { class: "rp-arrows", viewBox: "0 0 100 100", "aria-hidden": "true" });
  const defs = s("defs", null);
  for (const k of ARROW_KINDS) {
    defs.append(s("marker", { id: "arw-" + k, class: "rp-head rp-head--" + k, markerUnits: "userSpaceOnUse",
      markerWidth: "4.5", markerHeight: "4.5", refX: "2.4", refY: "2", orient: "auto", viewBox: "0 0 4 4" },
      s("path", { d: "M0,0 L4,2 L0,4 z" })));
  }
  arrows.append(defs);
  const seatLayer = h("div", { class: "rp-seats" });
  const seatEls = players.map((p, i) => {
    const el = h("button", { class: "seat", type: "button", style: "left:" + pos[i].x + "%;top:" + pos[i].y + "%" },
      h("span", { class: "seat__role role role--" + factionOf(p.role), text: p.role }),
      h("span", { class: "seat__name", text: p.name }),
    );
    el.addEventListener("click", () => { st.sel = st.sel === p.name ? null : p.name; paintDrawer(); paintTable(); });
    seatLayer.append(el);
    return el;
  });
  const table = h("div", { class: "rp-table" }, h("div", { class: "rp-hearth", "aria-hidden": "true" }, h("span", { class: "rp-hearth__glyph" })), arrows, seatLayer);

  // ---- feed (with a reasoning ticker that surfaces the acting agent's thought) ----
  const now = h("button", { class: "rp-now", type: "button" });
  now.addEventListener("click", () => { const a = events[st.cursor].actor; if (a) { st.sel = a; paintDrawer(); paintTable(); } });
  const feed = h("div", { class: "rp-feed" });
  const stage = h("div", { class: "rp-stage" }, h("div", { class: "rp-tablewrap" }, table), h("div", { class: "rp-feedcol" }, now, feed));

  // ---- transport ----
  const btnPrev = h("button", { class: "rp-btn", type: "button", "aria-label": "Step back", text: "◀" });
  const btnPlay = h("button", { class: "rp-btn rp-btn--play", type: "button", "aria-label": "Play" });
  const btnNext = h("button", { class: "rp-btn", type: "button", "aria-label": "Step forward", text: "▶" });
  const btnSpeed = h("button", { class: "rp-btn rp-btn--speed", type: "button", "aria-label": "Playback speed", text: "1×" });
  const progress = h("div", { class: "rp-progress", "aria-hidden": "true" }, h("div", { class: "rp-progress__bar" }));
  const counter = h("span", { class: "rp-counter" });
  const range = h("input", { class: "rp-range", type: "range", min: "0", max: String(events.length - 1), value: "0", "aria-label": "Timeline position" });
  const segs = h("div", { class: "rp-segs" });
  g.phases.forEach((ph) => {
    const span = ph.last_seq - ph.first_seq + 1;
    const seg = h("button", { class: "rp-seg", type: "button", style: "flex:" + span + " 1 0", title: ph.label },
      h("span", { class: "rp-seg__glyph", "aria-hidden": "true", text: PHASE_GLYPH[ph.phase] }),
      h("span", { class: "rp-seg__label", text: ph.label }));
    seg.addEventListener("click", () => { pause(); setCursor(ph.first_seq); });
    segs.append(seg);
  });
  const transport = h("div", { class: "rp-transport" },
    h("div", { class: "rp-controls" }, btnPrev, btnPlay, btnNext, btnSpeed, counter),
    progress,
    h("div", { class: "rp-scrub" }, segs, range));

  const drawer = h("aside", { class: "rp-drawer", "aria-label": "Player detail", "aria-hidden": "true" });

  // Same search/filter as the dashboard, so you can switch games without going back.
  const browse = h("details", { class: "rp-browse" },
    h("summary", { class: "rp-browse__summary" },
      h("span", { class: "rp-browse__label", text: "Browse games" }),
      h("span", { class: "rp-browse__hint", "aria-hidden": "true", text: "▾" })),
    DATA && DATA.games
      ? gamesBrowser(DATA.games, g.game_id)
      : h("p", { class: "panel__note", text: "Game list unavailable." }));

  root.replaceChildren(topbar, browse, stage, transport, drawer);

  // ---- transport behavior ----
  // Dwell scales with how much there is to read, so a long speech lingers and a
  // one-line resolution does not overstay; speed divides it. The progress bar
  // shows the wait is intentional, not a stall.
  const SPEEDS = [1, 1.5, 2, 0.5];
  const bar = progress.firstChild;
  function dwellFor(e) {
    const n = (str) => (typeof str === "string" ? str.length : 0);
    let ms;
    switch (e.type) {
      case "speech": ms = 1300 + n(e.payload.message) * 15; break;
      case "werewolf_chat": ms = 1200 + n(e.payload.message) * 15; break;
      case "accusation": case "defense": ms = 1700 + n(e.payload.reason) * 11; break;
      case "seer_inspect": case "doctor_protect": ms = 1700; break;
      case "kill_ballots": ms = 2100; break;
      case "discussion_resolved": ms = 2600; break;
      case "kill_resolved": case "exile_resolved": ms = 2900; break;
      case "game_over": ms = 3200; break;
      default: ms = 1400;
    }
    return Math.max(1400, Math.min(7000, ms)) / st.speed;
  }
  function clearTimer() { if (st.timer) { clearTimeout(st.timer); st.timer = null; } }
  function resetProgress() { bar.style.transition = "none"; bar.style.width = "0%"; progress.classList.remove("is-running"); }
  function runProgress(ms) {
    progress.classList.add("is-running");
    bar.style.transition = "none"; bar.style.width = "0%";
    void bar.offsetWidth; // reflow so 0% lands before animating to 100%
    bar.style.transition = "width " + ms + "ms linear"; bar.style.width = "100%";
  }
  function setCursor(c) { st.cursor = Math.max(0, Math.min(events.length - 1, c)); paint(); }
  function schedule() {
    clearTimer();
    if (!st.playing) { resetProgress(); return; }
    if (st.cursor >= events.length - 1) { pause(); return; }
    const ms = dwellFor(events[st.cursor]);
    runProgress(ms);
    st.timer = setTimeout(() => { st.cursor += 1; paint(); schedule(); }, ms);
  }
  function play() { if (st.cursor >= events.length - 1) setCursor(0); st.playing = true; updatePlay(); schedule(); }
  function pause() { st.playing = false; clearTimer(); resetProgress(); updatePlay(); }
  function updatePlay() {
    btnPlay.textContent = st.playing ? "‖" : "▶"; // pause bars / play
    btnPlay.setAttribute("aria-label", st.playing ? "Pause" : "Play");
    btnPlay.classList.toggle("is-playing", st.playing);
  }
  btnPrev.addEventListener("click", () => { pause(); setCursor(st.cursor - 1); });
  btnNext.addEventListener("click", () => { pause(); setCursor(st.cursor + 1); });
  btnPlay.addEventListener("click", () => (st.playing ? pause() : play()));
  btnSpeed.addEventListener("click", () => {
    st.speed = SPEEDS[(SPEEDS.indexOf(st.speed) + 1) % SPEEDS.length];
    btnSpeed.textContent = (st.speed === 1.5 ? "1.5" : String(st.speed)) + "×";
    if (st.playing) schedule();
  });
  range.addEventListener("input", () => { pause(); setCursor(Number(range.value)); });

  // ---- painters ----
  function curBlock() { return g.phases.find((p) => events[st.cursor].seq >= p.first_seq && events[st.cursor].seq <= p.last_seq); }

  function paint() {
    const e = events[st.cursor];
    const block = curBlock();
    phase.replaceChildren(h("span", { class: "rp-phase__glyph", "aria-hidden": "true", text: PHASE_GLYPH[e.phase] }),
      h("b", { text: block ? block.label : e.phase }));
    range.value = String(st.cursor);
    counter.textContent = (st.cursor + 1) + " / " + events.length;
    [...segs.children].forEach((seg, i) => {
      const p = g.phases[i];
      seg.classList.toggle("rp-seg--active", e.seq >= p.first_seq && e.seq <= p.last_seq);
    });
    updatePlay();
    paintNow();
    paintTable();
    paintFeed();
    paintDrawer();
  }

  function decisionForEvent(e) {
    if (!e.actor) return null;
    let exact = null;
    let latest = null;
    for (const t of g.trajectories) {
      if (t.caller !== e.actor) continue;
      if (t.anchor_seq === e.seq) exact = t;
      if (t.anchor_seq <= st.cursor && (!latest || t.anchor_seq > latest.anchor_seq)) latest = t;
    }
    return exact || latest;
  }

  function paintNow() {
    const e = events[st.cursor];
    const block = curBlock();
    const dec = decisionForEvent(e);
    let thought = "";
    if (dec) {
      for (const stp of dec.react_trajectory) if (stp.thought) thought = stp.thought;
    }
    now.classList.toggle("rp-now--has", !!e.actor);
    now.replaceChildren(
      h("span", { class: "rp-now__phase" }, h("span", { class: "rp-now__glyph", "aria-hidden": "true", text: PHASE_GLYPH[e.phase] }), block ? block.label : e.phase),
      e.actor
        ? h("span", { class: "rp-now__who" },
            h("span", { class: "rp-now__name role--" + factionOf(roleByName[e.actor] || "villager"), text: e.actor }),
            h("span", { class: "rp-now__thought", text: thought ? "“" + thought + "”" : "is acting…" }))
        : h("span", { class: "rp-now__who rp-now__who--idle", text: "the table resolves" }),
    );
  }

  function paintTable() {
    const e = events[st.cursor];
    table.classList.toggle("rp-table--day", e.phase === "day");
    const hg = table.querySelector(".rp-hearth__glyph");
    if (hg) hg.textContent = PHASE_GLYPH[e.phase];
    const dead = deadThrough(events, st.cursor);
    const deadBefore = deadThrough(events, st.cursor - 1);
    const target = e.type === "kill_resolved" ? e.payload.victim : e.type === "exile_resolved" ? e.payload.exiled : null;
    seatEls.forEach((el, i) => {
      const name = players[i].name;
      el.classList.toggle("seat--dead", dead.has(name));
      el.classList.toggle("seat--active", e.actor === name);
      el.classList.toggle("seat--target", name === target);
      el.classList.toggle("seat--selected", st.sel === name);
    });
    [...arrows.querySelectorAll(".rp-arrow")].forEach((n) => n.remove());
    for (const ar of arrowsFor(e, players, deadBefore)) {
      const a = pos[idxByName[ar.from]];
      const b = pos[idxByName[ar.to]];
      if (!a || !b) continue;
      const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len, pad = 9;
      arrows.append(s("line", { class: "rp-arrow rp-arrow--" + ar.kind, "marker-end": "url(#arw-" + ar.kind + ")", pathLength: "1",
        x1: a.x + ux * pad, y1: a.y + uy * pad, x2: b.x - ux * (pad + 1), y2: b.y - uy * (pad + 1) }));
    }
  }

  function paintFeed() {
    feed.replaceChildren();
    let lastLabel = null;
    for (let i = 0; i <= st.cursor; i++) {
      const e = events[i];
      const block = g.phases.find((ph) => e.seq >= ph.first_seq && e.seq <= ph.last_seq);
      const label = block ? block.label : e.phase;
      if (label !== lastLabel) { feed.append(phaseDivider(label, e.phase)); lastLabel = label; }
      const item = feedItem(e, i === st.cursor);
      if (item) feed.append(item);
    }
    const cur = feed.querySelector(".is-current");
    if (cur) {
      const top = cur.offsetTop, bot = top + cur.offsetHeight;
      if (top < feed.scrollTop) feed.scrollTop = top - 10;
      else if (bot > feed.scrollTop + feed.clientHeight) feed.scrollTop = bot - feed.clientHeight + 10;
    }
  }

  function feedItem(e, current) {
    const base = "feed-item" + (current ? " is-current" : "");
    const p = e.payload;
    switch (e.type) {
      case "speech": return say(base, e.actor, p.message, null);
      case "werewolf_chat": return say(base + " feed-item--pack", e.actor, p.message, "pack");
      case "accusation": return act(base + " feed-item--accuse", e.actor, "accuses", p.target, p.reason);
      case "defense": return act(base + " feed-item--defend", e.actor, "defends", p.defended, p.reason);
      case "seer_inspect": return secret(base + " feed-item--seer", e.actor + " inspects " + p.target, p.target + " is " + p.faction);
      case "doctor_protect": return secret(base + " feed-item--doctor", e.actor + " guards " + p.target, null);
      case "kill_ballots": return killBallots(base + " feed-item--pack", p, current);
      case "discussion_resolved": return bidPanel(base + " feed-item--bids", p, current);
      case "kill_resolved": return stageLine(base + " feed-item--kill", p.victim ? p.victim + " is found dead at dawn." : "The night passes; everyone survives.");
      case "exile_resolved": return votePanel(base + " feed-item--vote", p, current);
      case "game_over": return stageLine(base + " feed-item--over", (p.winner === "werewolves" ? "The werewolves" : "The village") + " win.");
      default: return null; // bid, tool_rejected live in the seat drawer
    }
  }

  function say(cls, name, message, tag) {
    const fac = factionOf(roleByName[name] || "villager");
    return h("div", { class: cls + " feed-item--say feed-item--" + fac },
      h("div", { class: "say__head" },
        h("span", { class: "say__name", text: name }),
        tag ? h("span", { class: "say__tag", text: tag }) : null),
      h("p", { class: "say__msg", text: message }));
  }
  function act(cls, who, verb, whom, reason) {
    return h("div", { class: cls + " feed-item--act" },
      h("p", { class: "act__line" }, h("b", { text: who }), " " + verb + " ", h("b", { text: whom })),
      reason ? h("p", { class: "act__reason", text: reason }) : null);
  }
  function secret(cls, head, detail) {
    return h("div", { class: cls + " feed-item--secret" },
      h("span", { class: "secret__head", text: head }),
      detail ? h("span", { class: "secret__detail", text: detail }) : null);
  }
  function stageLine(cls, text) {
    return h("div", { class: cls + " feed-item--stage" }, h("span", { class: "stage__text", text: text }));
  }

  function phaseDivider(label, phase) {
    return h("div", { class: "feed-divider feed-divider--" + phase },
      h("span", { class: "feed-divider__glyph", "aria-hidden": "true", text: PHASE_GLYPH[phase] }),
      h("span", { class: "feed-divider__label", text: label }));
  }

  function tallyBar(frac, win, anim) {
    return h("span", { class: "vote-row__bar" },
      h("span", { class: "vote-row__fill" + (win ? " vote-row__fill--win" : "") + (anim ? " bar-anim" : ""), style: "width:" + Math.round(frac * 100) + "%" }));
  }

  function bidPanel(cls, p, current) {
    const bids = p.bids || {};
    const winners = new Set(p.speakers || []);
    const rows = Object.entries(bids).sort((a, b) => b[1] - a[1]);
    const max = Math.max(1, ...rows.map((r) => r[1]));
    const list = h("div", { class: "vote__rows" });
    for (const [name, amt] of rows) {
      const win = winners.has(name);
      list.append(h("div", { class: "vote-row" + (win ? " vote-row--win" : "") },
        h("span", { class: "vote-row__who", text: name }),
        tallyBar(amt / max, win, current),
        h("span", { class: "vote-row__count", text: String(amt) }),
        win ? h("span", { class: "vote-row__tag", text: "speaks" }) : null));
    }
    return h("div", { class: cls + " feed-item--panel" },
      h("div", { class: "panel-row__head" }, h("span", { class: "panel-row__title", text: "Bidding for the floor" }),
        h("span", { class: "panel-row__out", text: (p.speakers || []).join(" › ") })),
      list);
  }

  function votePanel(cls, p, current) {
    const ballots = p.ballots || {};
    const tally = {};
    for (const [voter, target] of Object.entries(ballots)) {
      const key = !target || target === "abstain" ? "abstain" : target;
      (tally[key] = tally[key] || []).push(voter);
    }
    const rows = Object.entries(tally).sort((a, b) => b[1].length - a[1].length);
    const max = Math.max(1, ...rows.map((r) => r[1].length));
    const list = h("div", { class: "vote__rows" });
    for (const [target, voters] of rows) {
      const out = target === p.exiled;
      list.append(h("div", { class: "vote-row" + (out ? " vote-row--out" : "") + (target === "abstain" ? " vote-row--abstain" : "") },
        h("span", { class: "vote-row__who", text: target }),
        tallyBar(voters.length / max, out, current),
        h("span", { class: "vote-row__count", text: String(voters.length) }),
        h("span", { class: "vote-row__voters", text: voters.join(", ") })));
    }
    return h("div", { class: cls + " feed-item--panel" },
      h("div", { class: "panel-row__head" }, h("span", { class: "panel-row__title", text: "The village votes" }),
        h("span", { class: "panel-row__out", text: p.exiled ? p.exiled + " is exiled" : "tie · no exile" })),
      list);
  }

  function killBallots(cls, p, current) {
    const tally = {};
    for (const [wolf, target] of Object.entries(p.ballots || {})) (tally[target] = tally[target] || []).push(wolf);
    const rows = Object.entries(tally).sort((a, b) => b[1].length - a[1].length);
    const max = Math.max(1, ...rows.map((r) => r[1].length));
    const list = h("div", { class: "vote__rows" });
    for (const [target, wolves] of rows) {
      list.append(h("div", { class: "vote-row vote-row--out" },
        h("span", { class: "vote-row__who", text: target }),
        tallyBar(wolves.length / max, true, current),
        h("span", { class: "vote-row__count", text: String(wolves.length) }),
        h("span", { class: "vote-row__voters", text: wolves.join(", ") })));
    }
    return h("div", { class: cls + " feed-item--panel feed-item--secret" },
      h("span", { class: "secret__head", text: "The pack marks its prey" }), list);
  }

  function currentDecision(name) {
    let best = null;
    for (const t of g.trajectories) {
      if (t.caller !== name || t.anchor_seq > st.cursor) continue;
      if (!best || t.anchor_seq > best.anchor_seq || (t.anchor_seq === best.anchor_seq && t.decision_seq > best.decision_seq)) best = t;
    }
    return best;
  }
  function currentMemory(name) {
    let best = null;
    for (const m of g.memory_timeline) {
      if (m.player !== name || m.at_seq > st.cursor) continue;
      if (!best || m.at_seq >= best.at_seq) best = m;
    }
    return best;
  }

  function paintDrawer() {
    if (!st.sel) { drawer.classList.remove("rp-drawer--open"); drawer.setAttribute("aria-hidden", "true"); drawer.replaceChildren(); return; }
    const name = st.sel;
    const role = roleByName[name];
    const player = players[idxByName[name]];
    const dead = deadThrough(events, st.cursor).has(name);
    const dec = currentDecision(name);
    const mem = currentMemory(name);
    const round = events[st.cursor].round;
    const fm = (g.final_memories && g.final_memories[name]) || { plan: "", beliefs: {}, notes: [] };
    const notes = (fm.notes || []).filter((n) => n.round <= round);
    const beliefs = mem ? mem.beliefs : {};
    const plan = mem ? mem.plan : "";

    const close = h("button", { class: "drawer__close", type: "button", "aria-label": "Close", text: "×" });
    close.addEventListener("click", () => { st.sel = null; paintDrawer(); paintTable(); });

    const sections = [];
    if (dec) {
      const tok = dec.lm_calls.reduce((s2, c) => s2 + (c.prompt_tokens || 0) + (c.completion_tokens || 0), 0);
      const cost = dec.lm_calls.reduce((s2, c) => s2 + (c.cost_usd || 0), 0);
      const steps = h("div", { class: "react" });
      dec.react_trajectory.forEach((stp) => {
        steps.append(h("div", { class: "react-step" },
          stp.thought ? h("p", { class: "react-step__thought", text: stp.thought }) : null,
          h("p", { class: "react-step__act" },
            h("span", { class: "react-step__tool", text: stp.tool }),
            h("span", { class: "react-step__args", text: argLine(stp.args) })),
          stp.observation ? h("p", { class: "react-step__obs", text: stp.observation }) : null));
      });
      sections.push(h("div", { class: "drawer__section" },
        h("h3", { class: "drawer__h", text: "Reasoning · " + (curBlock() ? curBlock().label : "") }),
        steps,
        h("p", { class: "telemetry" }, h("b", { text: int(tok) }), " tokens", cost ? h("span", null, "  ·  $" + cost.toFixed(4)) : null)));
    } else {
      sections.push(h("div", { class: "drawer__section" }, h("p", { class: "drawer__empty", text: "No move from " + name + " yet at this point." })));
    }

    const subjects = Object.keys(beliefs).sort();
    if (plan || subjects.length) {
      const body = [];
      if (plan) body.push(h("p", { class: "plan", text: plan }));
      if (subjects.length) {
        const tbl = h("div", { class: "beliefs" });
        for (const subj of subjects) {
          const b = beliefs[subj];
          tbl.append(h("div", { class: "belief" },
            h("span", { class: "belief__subj", text: subj }),
            h("span", { class: "belief__guess belief__guess--" + factionOf(b.guess === "werewolf" ? "werewolf" : "village"), text: b.guess }),
            h("span", { class: "conf conf--" + b.confidence, text: b.confidence }),
            h("p", { class: "belief__evidence", text: b.evidence })));
        }
        body.push(tbl);
      }
      sections.push(h("div", { class: "drawer__section" }, h("h3", { class: "drawer__h", text: "What " + name + " believes" }), ...body));
    }

    if (notes.length) {
      const nl = h("ul", { class: "notes" });
      for (const n of notes) nl.append(h("li", { class: "note" }, h("span", { class: "note__round", text: "R" + n.round }), n.text));
      sections.push(h("div", { class: "drawer__section" }, h("h3", { class: "drawer__h", text: "Notes" }), nl));
    }

    drawer.replaceChildren(
      h("div", { class: "drawer__head" },
        h("div", { class: "drawer__id" },
          h("span", { class: "role role--" + factionOf(role), text: role }),
          h("span", { class: "drawer__name", text: name + (dead ? " (dead)" : "") }),
          h("span", { class: "drawer__model" }, modelEl(player.model || "unknown"))),
        close),
      ...sections);
    drawer.classList.add("rp-drawer--open");
    drawer.setAttribute("aria-hidden", "false");
  }

  function argLine(args) {
    const parts = [];
    for (const [k, v] of Object.entries(args || {})) {
      let val = typeof v === "string" ? v : JSON.stringify(v);
      if (val.length > 80) val = val.slice(0, 79) + "…";
      parts.push(k + ": " + val);
    }
    return parts.length ? "(" + parts.join(", ") + ")" : "";
  }

  paint();
  return { pause };
}

// --- boot & routing ------------------------------------------------------

function showError(message) {
  const banner = document.getElementById("error");
  if (banner) { banner.textContent = message; banner.hidden = false; }
}

function render(data) {
  renderDateline(data.meta);
  renderLeaderboard(data.leaderboard);
  renderGames(data.games);
  renderScatter(data.deceiver_detector);
  renderHeadToHead(data.head_to_head);
  renderCost(data.cost_efficiency);
  renderSelfPlay(data.self_play);
}

let DATA = null;
let dashRendered = false;
let currentReplay = null;

function showDashboard() {
  if (currentReplay) { currentReplay.pause(); currentReplay = null; }
  document.getElementById("replay").hidden = true;
  document.getElementById("main").hidden = false;
  document.querySelector(".footer").hidden = false;
  if (DATA && !dashRendered) { render(DATA); dashRendered = true; }
  window.scrollTo(0, 0);
}

function showReplay(id) {
  if (currentReplay) { currentReplay.pause(); currentReplay = null; }
  if (!/^g[\w.\-]+$/.test(id)) { location.hash = "#/"; return; }
  document.getElementById("main").hidden = true;
  document.querySelector(".footer").hidden = true;
  const root = document.getElementById("replay");
  root.hidden = false;
  root.replaceChildren(h("p", { class: "rp-loading", text: "Loading game…" }));
  window.scrollTo(0, 0);
  fetch("games/" + encodeURIComponent(id) + ".json")
    .then((r) => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then((g) => { if (location.hash.indexOf(id) >= 0) currentReplay = mountReplay(g); })
    .catch((err) => {
      root.replaceChildren(h("div", { class: "rp-loading" },
        h("p", { text: "Couldn't load this game (" + err.message + ")." }),
        h("a", { href: "#/", text: "← Back to replays" })));
    });
}

function route() {
  const m = location.hash.match(/^#\/game\/(.+)$/);
  if (m) showReplay(decodeURIComponent(m[1]));
  else showDashboard();
}

window.addEventListener("hashchange", route);

fetch('data.json')
  .then((r) => {
    if (!r.ok) throw new Error("HTTP " + r.status);
    return r.json();
  })
  .then((data) => {
    DATA = data;
    const fid = featuredGameId(data.games);
    const cta = document.getElementById("watch-cta");
    if (cta && fid) cta.href = "#/game/" + encodeURIComponent(fid);
    route();
  })
  .catch((err) => {
    showError(
      "Couldn't load data.json (" + err.message + "). This page needs a local server: run " +
      "python -m http.server  in this folder, then open the address it prints.",
    );
  });
