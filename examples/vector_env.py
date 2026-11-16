"""Four copies of a HiGym task in a Gymnasium vector environment.

    python examples/vector_env.py [task]    # default: cheetah-8_run
"""

import sys

import gymnasium

import higym

name = sys.argv[1] if len(sys.argv) > 1 else "cheetah-8_run"
envs = gymnasium.vector.SyncVectorEnv([lambda: higym.make(name) for _ in range(4)])
obs, info = envs.reset(seed=0)
for _ in range(100):
    obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
print(f"{name}: observations {obs.shape}, mean reward on the last step {reward.mean():.3f}")
envs.close()
