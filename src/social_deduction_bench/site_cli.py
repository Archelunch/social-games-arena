"""`sdb-site` CLI: build the static results site from tournament run dirs (T28).

Globs one or more run directories of per-game artifacts, re-aggregates them into the
deterministic `data.json` payload, and writes the self-contained static site. Fails
loud (exit 2, nothing written) when no game directory is usable, so an empty or
all-in-progress run never yields a misleading empty site.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from social_deduction_bench.games.werewolf.replay_data import write_replays
from social_deduction_bench.games.werewolf.site_data import build_site_data
from social_deduction_bench.site import write_site

_DEFAULT_RUN = "games/run2"
_DEFAULT_OUT = "site"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sdb-site",
        description="Build the static Werewolf benchmark results site from tournament run dirs.",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        metavar="DIR",
        help=f"run directory of per-game artifacts (repeatable; default: {_DEFAULT_RUN})",
    )
    parser.add_argument(
        "--out",
        default=_DEFAULT_OUT,
        metavar="DIR",
        help=f"output directory for the generated site (default: {_DEFAULT_OUT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build the site. Returns 0 on success, 2 when no usable game directory is found."""
    args = _build_parser().parse_args(argv)
    run_dirs = [Path(p) for p in (args.run or [_DEFAULT_RUN])]
    out_dir = Path(args.out)

    try:
        data = build_site_data(run_dirs)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    write_site(data, out_dir)
    n_replays = write_replays(run_dirs, out_dir)
    meta = data["meta"]
    print(f"site: {meta['n_games']} games ({meta['n_rated']} rated) -> {out_dir}")
    print(f"  replays: {n_replays} game files -> {out_dir}/games")
    print(f"  serve it:  (cd {out_dir} && python -m http.server)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
