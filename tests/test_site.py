"""Tests for the static-site writer and the front-end asset contracts (T28).

`write_site` writes the deterministic `data.json` plus the three static assets that
render it. The asset-contract tests assert the *output* artifact upholds the design's
hard rules: it must work fully offline (no external resource refs, browsers block
`file://` fetch only for cross-origin, but a CDN ref would break a no-network viewer),
degrade gracefully (`<noscript>`, viewport, `lang`), follow the OKLCH/tinted-neutral
color rule (never pure black/white), and be DOM-safe (model names contain `/`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from social_deduction_bench.site import write_site


def _written(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    data: dict[str, object] = {"meta": {"n_games": 1}, "leaderboard": []}
    out = tmp_path / "site"
    write_site(data, out)
    return out, data


def test_write_site_emits_data_json_and_assets(tmp_path: Path) -> None:
    out, data = _written(tmp_path)
    assert (out / "data.json").exists()
    assert (out / "index.html").exists()
    assert (out / "styles.css").exists()
    assert (out / "app.js").exists()
    assert json.loads((out / "data.json").read_text()) == data


def test_write_site_data_json_is_byte_stable(tmp_path: Path) -> None:
    """`write_site` must emit byte-identical data.json for equal payloads (sort_keys).

    Asserts the RAW serialized text, not a json.loads round-trip (which is order-blind),
    so a regression that drops `sort_keys` in write_site would be caught.
    """
    data: dict[str, object] = {"b": 1, "a": {"y": 2, "x": 3}, "list": [3, 1, 2]}
    write_site(data, tmp_path / "one")
    write_site(data, tmp_path / "two")
    first = (tmp_path / "one" / "data.json").read_bytes()
    second = (tmp_path / "two" / "data.json").read_bytes()
    assert first == second
    # sort_keys contract: nested keys are alphabetized in the raw text.
    text = first.decode()
    assert text.index('"a"') < text.index('"b"')
    assert text.index('"x"') < text.index('"y"')


def _external_refs(source: str) -> list[str]:
    """External resource loads (CDNs, web fonts) that would break offline use.

    The SVG/XML namespace URI (w3.org) is a constant identifier, not a network fetch,
    so it is not an external dependency.
    """
    return [m for m in re.findall(r"https?://[^\s\"')]+", source) if "w3.org" not in m]


def test_index_html_has_no_external_refs_and_references_assets(tmp_path: Path) -> None:
    out, _ = _written(tmp_path)
    html = (out / "index.html").read_text()
    js = (out / "app.js").read_text()
    assert _external_refs(html) == [], "index.html must have no external resource refs"
    assert _external_refs(js) == [], "app.js must have no external resource refs"
    assert "data.json" in (html + js)  # the payload is fetched
    assert "styles.css" in html
    assert "app.js" in html


def test_index_html_has_viewport_lang_and_noscript(tmp_path: Path) -> None:
    out, _ = _written(tmp_path)
    html = (out / "index.html").read_text()
    assert 'name="viewport"' in html
    assert 'lang="en"' in html
    assert "<noscript" in html.lower()


def test_styles_css_uses_oklch_tokens_no_pure_black_white(tmp_path: Path) -> None:
    out, _ = _written(tmp_path)
    css = (out / "styles.css").read_text().lower()
    assert "oklch(" in css
    for token in ("--ground", "--ink", "--wolf", "--village", "--gilt"):
        assert token in css, f"missing design token {token}"
    # "#000"/"#fff" substrings also catch "#000000"/"#ffffff".
    for banned in ("#000", "#fff"):
        assert banned not in css, f"pure black/white hex {banned} is banned (tint neutrals)"
    flat = css.replace(" ", "")
    # flat has spaces stripped: pure black "oklch(0 0 0" -> "oklch(000", white -> "oklch(100".
    for banned in ("rgb(0,0,0)", "rgb(255,255,255)", "oklch(000", "oklch(100"):
        assert banned not in flat, f"pure black/white color {banned!r} is banned (tint neutrals)"


def test_app_js_is_offline_and_dom_safe(tmp_path: Path) -> None:
    out, _ = _written(tmp_path)
    js = (out / "app.js").read_text()
    assert "fetch('data.json')" in js or 'fetch("data.json")' in js
    assert "prefers-reduced-motion" in js  # motion respects the OS setting
    assert "data-model" in js  # names reach the DOM via data attributes
    assert "textContent" in js  # ... and textContent, never innerHTML
    # Model names contain "/": never build an element id / selector from one.
    assert "getElementById(model" not in js
    flat_js = js.replace(" ", "")
    assert 'querySelector("#"+' not in flat_js
    assert "querySelector('#'+" not in flat_js
