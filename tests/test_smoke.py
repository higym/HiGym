"""Every public id constructs, resets and steps."""

import numpy as np
import pytest

import higym


def _param(name):
    dof = int(name.split("-", 1)[1].split("_", 1)[0])
    return pytest.param(name, marks=pytest.mark.slow) if dof >= 128 else name


@pytest.mark.parametrize("name", [_param(n) for n in higym.list_tasks()])
def test_every_id_resets_and_steps(name):
    env = higym.make(name)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs) and np.isfinite(reward)
    assert (terminated, truncated) == (False, False)
    env.close()
