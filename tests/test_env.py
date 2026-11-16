"""HiGymEnv reproduces the paper's wrapper stack and the Gymnasium contract."""

import gymnasium
import numpy as np
import pytest
from dm_control.suite.wrappers import action_scale

import higym
from higym._suite import DOMAINS, custom_tasks
from higym.env import HiGymEnv


def _env(**kwargs):
    return HiGymEnv("cheetah", "8_run", **kwargs)


def _actions(env, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, (n, *env.action_space.shape)).astype(np.float32)


def _flat(obs):
    return np.concatenate([v.flatten() for v in obs.values()], dtype=np.float32)


def test_spaces_and_dtypes():
    env = _env()
    obs, info = env.reset(seed=0)
    assert env.observation_space.dtype == np.float32
    assert obs.dtype == np.float32 and obs.shape == env.observation_space.shape
    assert env.action_space.dtype == np.float32
    assert env.action_space.shape == env.dm_env.action_spec().shape
    assert np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1)
    assert info == {"score": 0.0}
    assert env.metadata["render_fps"] == 50  # cheetah: 0.01 s control step, repeat 2


@pytest.mark.parametrize("repeat", [1, 2, 3, 4])
def test_episode_is_1000_control_steps(repeat):
    env = _env(action_repeat=repeat)
    assert env.max_episode_steps == 1000 // repeat
    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape, np.float32)
    for t in range(1, 1000 // repeat + 1):
        _, _, terminated, truncated, _ = env.step(zero)
        assert terminated is False
        assert truncated == (t == 1000 // repeat)


@pytest.mark.slow
@pytest.mark.parametrize("domain,task", [(d, t) for d in DOMAINS for t in custom_tasks(d)])
def test_every_task_has_a_1000_step_limit(domain, task):
    assert higym.load(domain, task)._step_limit == 1000


def test_matches_paper_stack_after_reset_seed():
    # The paper built suite.load(seed) -> action_scale -> repeat-2 summing rewards.
    env = _env()
    obs, _ = env.reset(seed=7)
    ref = action_scale.Wrapper(higym.load("cheetah", "8_run", seed=7), minimum=-1.0, maximum=1.0)
    assert np.array_equal(obs, _flat(ref.reset().observation))
    total = 0
    for action in _actions(env, 50):
        obs, reward, _, _, info = env.step(action)
        ref_reward = 0
        for _ in range(2):
            ts = ref.step(action.astype(np.float64))
            ref_reward += ts.reward
        total += ref_reward
        assert np.array_equal(obs, _flat(ts.observation))
        assert reward == ref_reward
        assert info["score"] == total / 1000


def test_same_seed_same_episode():
    env = _env()
    actions = _actions(env, 20)

    def episode(seed):
        return np.stack([env.reset(seed=seed)[0]] + [env.step(a)[0] for a in actions])

    assert np.array_equal(episode(5), episode(5))
    assert not np.array_equal(episode(5), episode(6))


def test_unseeded_reset_continues_the_stream():
    env = _env()
    env.reset(seed=5)
    second, _ = env.reset()
    ref = higym.load("cheetah", "8_run", seed=5)
    ref.reset()
    assert np.array_equal(second, _flat(ref.reset().observation))
    assert not np.array_equal(second, env.reset()[0])


def test_step_before_reset_raises():
    env = _env()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(env.action_space.shape, np.float32))


def test_step_after_truncation_raises_until_reset():
    env = _env(action_repeat=250)  # 4 agent steps per episode
    env.reset(seed=0)
    zero = np.zeros(env.action_space.shape, np.float32)
    for _ in range(4):
        env.step(zero)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(zero)
    env.reset()
    env.step(zero)


@pytest.mark.parametrize("make_action", [
    lambda n: 0.5,
    lambda n: np.zeros(n - 1, np.float32),
    lambda n: np.zeros((1, n), np.float32),
])
def test_wrong_action_shape_raises(make_action):
    env = _env()
    env.reset(seed=0)
    with pytest.raises(ValueError, match="shape"):
        env.step(make_action(env.action_space.shape[0]))


def test_list_and_float64_actions_are_accepted():
    env = _env()
    env.reset(seed=0)
    n = env.action_space.shape[0]
    env.step([0.0] * n)
    env.step(np.zeros(n, np.float64))


@pytest.mark.parametrize("repeat", [0, -1, 2.0, True, 1001])
def test_invalid_action_repeat_raises(repeat):
    with pytest.raises(ValueError, match="action_repeat"):
        _env(action_repeat=repeat)


def test_gymnasium_make_forwards_kwargs():
    env = higym.make("cheetah-8_run", action_repeat=4)
    assert env.unwrapped.max_episode_steps == 250
    assert isinstance(env.unwrapped, HiGymEnv)


def test_render_without_mode_warns_and_returns_none():
    env = _env()
    env.reset(seed=0)
    with pytest.warns(UserWarning, match="render_mode"):
        assert env.render() is None


def test_unsupported_render_mode_raises():
    with pytest.raises(ValueError, match="render_mode"):
        _env(render_mode="human")


def test_render_rgb_array():
    env = _env(render_mode="rgb_array", width=48, height=64)
    env.reset(seed=0)
    frame = env.render()
    assert frame.shape == (64, 48, 3) and frame.dtype == np.uint8


def test_quadruped_renders_camera_2_by_default():
    def frame(**kwargs):
        env = HiGymEnv("quadruped", "16_walk", render_mode="rgb_array", width=32, height=32, **kwargs)
        env.reset(seed=0)
        return env.render()

    assert np.array_equal(frame(), frame(camera_id=2))
    assert not np.array_equal(frame(), frame(camera_id=0))
