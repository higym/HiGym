"""Opt-in ablation ids. Each check runs in a fresh interpreter, because
importing higym.ablations adds tasks to dm_control's suites and Gymnasium's
registry for the rest of the process, which the core tests must not see.
"""

import hashlib
import json
import pathlib
import subprocess
import sys

import pytest

TESTS_DIR = pathlib.Path(__file__).parent
ABLATION_FILE = TESTS_DIR.parent / "src" / "higym" / "tasks" / "ablation_scaling.py"
GOLDEN = json.loads((TESTS_DIR / "data" / "ablation_mjcf_sha256.json").read_text())
WALKER = ["64a25", "64a50", "64a100", "256a25", "256a50", "256a100", "64a00m", "256a00m", "256a10m",
          "64a25m", "64a50m", "64a100m", "256a00p", "8a100"]
REACHER = ["64a25", "64a50", "64a100", "16a25", "16a50", "16a100", "64s000", "64s500", "64s1000", "4s000"]
EXPECTED = [f"walker-{v}_run" for v in WALKER] + [f"reacher-{v}_hard" for v in REACHER]


def _run(code):
    out = subprocess.run([sys.executable, "-c", code, str(TESTS_DIR)], check=True, capture_output=True, text=True)
    return json.loads(out.stdout.splitlines()[-1])


def test_ablation_file_is_the_paper_file():
    # sha256 of highdim@e4fa0d5:tdmpc2/envs/tasks/ablation_scaling.py
    digest = hashlib.sha256(ABLATION_FILE.read_bytes()).hexdigest()
    assert digest == "41046bcb3ea958db447ed6b161079f3cb56628fa88472176d81d56e236a5de03"


def test_ids_registered_only_on_import():
    result = _run(
        "import json, gymnasium, higym\n"
        "before = {i for i in gymnasium.registry if i.startswith('higym/')}\n"
        "import higym.ablations\n"
        "after = {i for i in gymnasium.registry if i.startswith('higym/')}\n"
        "print(json.dumps({'added': sorted(after - before), 'ablations': higym.ablations.list_tasks(),\n"
        "                  'core': len(higym.list_tasks())}))"
    )
    assert result["ablations"] == EXPECTED
    assert result["added"] == sorted(f"higym/{n}-v0" for n in EXPECTED)
    assert result["core"] == 81


def test_ablation_mjcf_matches_golden():
    digests = _run(
        "import importlib, json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from helpers import mjcf_digest\n"
        "import higym.ablations\n"
        f"names = {sorted(GOLDEN)!r}\n"
        "out = {}\n"
        "for name in names:\n"
        "    domain, task = name.split('-', 1)\n"
        "    out[name] = mjcf_digest(importlib.import_module(f'dm_control.suite.{domain}').SUITE[task])\n"
        "print(json.dumps(out))"
    )
    assert digests == GOLDEN


def _smoke(names):
    return _run(
        "import json, higym.ablations, higym\n"
        f"for name in {names!r}:\n"
        "    env = higym.make(name)\n"
        "    env.reset(seed=0)\n"
        "    env.step(env.action_space.sample())\n"
        "print(json.dumps('ok'))"
    )


def test_ablation_ids_reset_and_step():
    assert _smoke([n for n in EXPECTED if "-256" not in n]) == "ok"


@pytest.mark.slow
def test_ablation_ids_reset_and_step_256():
    assert _smoke([n for n in EXPECTED if "-256" in n]) == "ok"
