"""Public ids, higym.load, and dm_control's own tasks staying untouched."""

import json
import pathlib
import subprocess
import sys

import gymnasium
import numpy as np
import pytest

import higym

TABLE1 = {
    "quadruped": ((16, 32, 64, 128, 256, 512), ("walk", "run", "escape", "fetch")),
    "cheetah": ((8, 16, 32, 64, 128, 256), ("run",)),
    "walker": ((8, 16, 32, 64, 128, 256), ("stand", "walk", "run")),
    "finger": ((2, 4, 8, 16, 32, 64), ("spin", "turn_easy", "turn_hard")),
    "reacher": ((2, 4, 8, 16, 32, 64), ("easy", "hard")),
    "humanoid_CMU": ((80,), ("stand", "walk", "run")),
}
EXPECTED = [f"{d}-{n}_{t}" for d, (sizes, tasks) in TABLE1.items() for n in sizes for t in tasks]
TESTS_DIR = str(pathlib.Path(__file__).parent)


def test_list_tasks_is_table1():
    assert len(EXPECTED) == 81
    assert higym.list_tasks() == EXPECTED


def test_exactly_the_table1_ids_are_registered():
    ids = {i for i in gymnasium.registry if i.startswith("higym/")}
    assert ids == {f"higym/{name}-v0" for name in EXPECTED}
    for name in EXPECTED:
        spec = gymnasium.spec(f"higym/{name}-v0")
        assert spec.entry_point == "higym.env:HiGymEnv"
        assert spec.max_episode_steps is None


@pytest.mark.parametrize("name", ["cheetah-4_run", "walker-4_stand", "humanoid_CMU-56_walk"])
def test_tasks_outside_table1_load_but_have_no_id(name):
    assert f"higym/{name}-v0" not in gymnasium.registry
    domain, task = name.split("-", 1)
    assert higym.load(domain, task).action_spec().shape


@pytest.mark.parametrize("domain,task", [("cheetah", "run"), ("cheetah", "3_run"), ("ant", "8_run")])
def test_load_rejects_non_higym_tasks(domain, task):
    with pytest.raises(ValueError, match="not a HiGym task"):
        higym.load(domain, task)


def test_load_seed_is_the_task_seed():
    def first_obs(seed):
        return higym.load("cheetah", "8_run", seed=seed).reset().observation

    a, b, c = first_obs(3), first_obs(3), first_obs(4)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert not all(np.array_equal(a[k], c[k]) for k in a)


_BUILTIN_PROBE = r"""
import hashlib, json, sys
sys.path.insert(0, sys.argv[2])
import numpy as np
if sys.argv[1] == "with":
    import higym  # noqa: F401
from dm_control import suite
from helpers import mjcf_digest

DOMAINS = ("quadruped", "cheetah", "walker", "finger", "reacher", "humanoid_CMU")
record = {"ALL_TASKS": [list(t) for t in suite.ALL_TASKS]}
for domain, task in suite.ALL_TASKS:
    if domain not in DOMAINS:
        continue
    record[f"{domain}-{task}/mjcf"] = mjcf_digest(lambda: suite.load(domain, task))
    env = suite.load(domain, task, task_kwargs={"random": 0})
    spec = env.action_spec()
    rng = np.random.default_rng(0)
    digest = hashlib.sha256()
    env.reset()
    for _ in range(20):
        ts = env.step(rng.uniform(spec.minimum, spec.maximum, spec.shape))
        digest.update(np.float64(ts.reward).tobytes())
        for v in ts.observation.values():
            digest.update(np.asarray(v, dtype=np.float64).tobytes())
    record[f"{domain}-{task}/rollout"] = digest.hexdigest()
print(json.dumps(record))
"""


def _probe(mode):
    out = subprocess.run(
        [sys.executable, "-c", _BUILTIN_PROBE, mode, TESTS_DIR],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def test_builtin_dm_control_tasks_untouched():
    without, with_higym = _probe("without"), _probe("with")
    assert len(without) > 10
    assert with_higym == without


def test_import_pulls_no_training_stack():
    code = (
        "import json, sys, higym; "
        "print(json.dumps([m for m in ('torch', 'hydra', 'wandb', 'tensordict') if m in sys.modules]))"
    )
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert json.loads(out) == []
