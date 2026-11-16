"""Shared test helpers."""

import hashlib
from unittest import mock

import numpy as np
from dm_control.mujoco import engine


class _Captured(Exception):
    pass


def _bytes(value):
    return value.encode() if isinstance(value, str) else bytes(value)


def mjcf_digest(factory):
    """SHA-256 of the MJCF string and assets a task factory hands to MuJoCo.

    ``Physics.from_xml_string`` is patched to record its arguments and abort,
    so the model is generated but never compiled.
    """
    captured = []

    def record(cls, xml_string, assets=None):
        captured.append((xml_string, assets or {}))
        raise _Captured

    with mock.patch.object(engine.Physics, "from_xml_string", classmethod(record)):
        try:
            factory()
        except _Captured:
            pass
    if len(captured) != 1:
        raise AssertionError(f"expected one from_xml_string call, got {len(captured)}")
    xml, assets = captured[0]
    digest = hashlib.sha256(_bytes(xml))
    for key in sorted(assets):
        digest.update(key.encode())
        digest.update(hashlib.sha256(_bytes(assets[key])).digest())
    return digest.hexdigest()


def reference_rollout(name, steps=50):
    """Observations and rewards of ``steps`` seeded random actions after ``reset(seed=0)``."""
    import higym

    env = higym.make(name)
    observations = [env.reset(seed=0)[0]]
    shape = (steps, *env.action_space.shape)
    actions = np.random.default_rng(0).uniform(-1, 1, shape).astype(np.float32)
    rewards = []
    for action in actions:
        obs, reward, *_ = env.step(action)
        observations.append(obs)
        rewards.append(reward)
    return np.stack(observations), np.array(rewards)
