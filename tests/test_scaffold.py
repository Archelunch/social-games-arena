"""Scaffold smoke tests.

These encode the T01 contract: the package and its game-agnostic subpackages
import cleanly, and the project runs on the Python version the design mandates
(3.13). They fail loudly if the package layout regresses or the Poetry env is
pinned to the wrong interpreter.
"""

import importlib
import sys

import pytest


def test_package_imports() -> None:
    """The top-level package must be importable from an installed env."""
    assert importlib.import_module("social_deduction_bench") is not None


@pytest.mark.parametrize("subpackage", ["engine", "games", "agents", "rating"])
def test_subpackages_import(subpackage: str) -> None:
    """Each game-agnostic subpackage from the CLAUDE.md layout must import."""
    module = importlib.import_module(f"social_deduction_bench.{subpackage}")
    assert module is not None


def test_python_version_is_313() -> None:
    """WEREWOLF_DESIGN.md and CLAUDE.md mandate Python 3.13.

    The default `python3` on this machine is 3.12; this guards against the
    Poetry env being created with the wrong interpreter.
    """
    assert sys.version_info[:2] == (3, 13)
