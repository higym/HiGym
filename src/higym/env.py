"""The paper's environment protocol as a Gymnasium environment."""

import numbers
from typing import Any

import gymnasium
import numpy as np
from dm_control.suite.wrappers import action_scale
from gymnasium import spaces

from higym._suite import load

# Every HiGym task runs 1000 control steps per episode.
_CONTROL_STEPS = 1000


class HiGymEnv(gymnasium.Env):
    """A HiGym task under the protocol the paper trained with.

    Actions in [-1, 1] are rescaled to each actuator's control range and
    repeated ``action_repeat`` times, summing the rewards. Observations are
    dm_control's observation dict flattened to float32 in dict order. An
    episode is 1000 control steps (``1000 // action_repeat`` agent steps),
    after which it is truncated; it never terminates. ``info["score"]`` is
    the episode return so far divided by 1000.

    ``dm_env`` is the underlying dm_control environment, before action
    scaling.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": None}

    def __init__(
        self,
        domain: str,
        task: str,
        action_repeat: int = 2,
        render_mode: str | None = None,
        width: int = 224,
        height: int = 224,
        camera_id: int | None = None,
    ):
        if (
            isinstance(action_repeat, bool)
            or not isinstance(action_repeat, numbers.Integral)
            or not 1 <= action_repeat <= _CONTROL_STEPS
        ):
            raise ValueError(f"action_repeat must be an int in [1, {_CONTROL_STEPS}], got {action_repeat!r}")
        if render_mode not in (None, "rgb_array"):
            raise ValueError(f"render_mode must be None or 'rgb_array', got {render_mode!r}")

        self.dm_env: Any = load(domain, task)
        self._env: Any = action_scale.Wrapper(self.dm_env, minimum=-1.0, maximum=1.0)
        self.action_repeat = int(action_repeat)
        self.max_episode_steps = _CONTROL_STEPS // self.action_repeat
        self.render_mode = render_mode
        self._width, self._height = width, height
        self._camera_id = (2 if domain == "quadruped" else 0) if camera_id is None else camera_id

        action_spec = self._env.action_spec()
        self._action_dtype = action_spec.dtype
        obs_dim = sum(int(np.prod(spec.shape)) for spec in self._env.observation_spec().values())
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, action_spec.shape, np.float32)
        fps = round(1 / (self.dm_env.control_timestep() * self.action_repeat))
        self.metadata = {**self.metadata, "render_fps": fps}

        self._needs_reset: bool = True
        self._steps: int = 0
        self._return: float = 0.0

    @staticmethod
    def _flatten(observation):
        return np.concatenate([v.flatten() for v in observation.values()], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            # Same state as the paper's suite.load(..., task_kwargs={"random": seed}).
            self.dm_env.task.random.seed(seed)
        self._needs_reset = False
        self._steps = 0
        self._return = 0.0
        return self._flatten(self._env.reset().observation), {"score": 0.0}

    def step(self, action):
        # dm_control would silently start a new episode here.
        if self._needs_reset:
            raise RuntimeError("call reset() before step(): the episode has not started or has ended")
        action = np.asarray(action, dtype=self._action_dtype)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, got {action.shape}")
        reward = 0.0
        for _ in range(self.action_repeat):
            timestep = self._env.step(action)
            reward += timestep.reward
        self._steps += 1
        self._return += reward
        truncated = self._steps >= self.max_episode_steps
        self._needs_reset = truncated
        return self._flatten(timestep.observation), float(reward), False, truncated, {"score": self._return / 1000}

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn("render() needs the env to be created with render_mode='rgb_array'")
            return None
        return self.dm_env.physics.render(self._height, self._width, camera_id=self._camera_id).copy()

    def close(self):
        self._env.close()
