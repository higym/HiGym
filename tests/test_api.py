"""Gymnasium API compliance and vector environments."""

import gymnasium
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import higym

SMALLEST = [
    "quadruped-16_walk", "cheetah-8_run", "walker-8_walk",
    "finger-2_spin", "reacher-2_easy", "humanoid_CMU-80_stand",
]


@pytest.mark.parametrize("name", SMALLEST)
def test_check_env(name):
    check_env(higym.make(name).unwrapped, skip_render_check=True)


def _make():
    return higym.make("cheetah-8_run")


@pytest.mark.parametrize("vector_env", [
    gymnasium.vector.SyncVectorEnv,
    lambda fns: gymnasium.vector.AsyncVectorEnv(fns, context="spawn"),
], ids=["sync", "async-spawn"])
def test_vector_env(vector_env):
    envs = vector_env([_make, _make])
    obs, _ = envs.reset(seed=[0, 1])
    assert obs.shape == (2, *envs.single_observation_space.shape)
    assert obs.dtype == np.float32
    assert not np.array_equal(obs[0], obs[1])
    for _ in range(3):
        obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
    assert reward.shape == (2,) and not terminated.any() and not truncated.any()
    assert info["score"].shape == (2,)
    envs.close()
