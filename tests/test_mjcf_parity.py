"""The generated MJCF of every task is pinned to the paper code's output."""

import importlib
import json
import pathlib

import pytest

from helpers import mjcf_digest

GOLDEN = json.loads((pathlib.Path(__file__).parent / "data" / "mjcf_sha256.json").read_text())
DOMAINS = ("quadruped", "cheetah", "walker", "finger", "reacher", "humanoid_CMU")


def _suite(domain):
    importlib.import_module(f"higym.tasks.{domain}")
    return importlib.import_module(f"dm_control.suite.{domain}").SUITE


def test_golden_covers_all_registered_tasks():
    registered = {f"{d}-{name}" for d in DOMAINS for name in _suite(d).tagged("custom")}
    assert registered == set(GOLDEN)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_mjcf_matches_golden(name):
    domain, task = name.split("-", 1)
    assert mjcf_digest(_suite(domain)[task]) == GOLDEN[name]
