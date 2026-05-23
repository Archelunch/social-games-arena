"""Static results-site writer (T28).

`write_site` serializes the pre-aggregated payload to `data.json` and copies the
bundled static assets (`index.html` / `styles.css` / `app.js`) alongside it, producing
a self-contained directory any static file server can host. The assets fetch
`data.json` at load — no framework, no build step, no network. The asset files live in
`site_assets/` next to this module and are read at write time (works for the editable
Poetry install this repo runs from).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

_ASSETS_DIR = Path(__file__).parent / "site_assets"
_ASSET_FILES = ("index.html", "styles.css", "app.js")


def write_site(data: dict[str, object], out_dir: Path) -> None:
    """Write the deterministic `data.json` and the static assets into `out_dir`.

    `data.json` is serialized with `sort_keys=True` so the same payload is always
    byte-identical (invariant #4); the assets are copied verbatim.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, sort_keys=True, indent=2) + "\n"
    (out_dir / "data.json").write_text(payload, encoding="utf-8")
    for name in _ASSET_FILES:
        shutil.copyfile(_ASSETS_DIR / name, out_dir / name)
