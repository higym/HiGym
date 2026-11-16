"""One episode of uniformly random actions on a HiGym task.

    python examples/random_policy.py [task]    # default: cheetah-8_run
"""

import sys

import higym

name = sys.argv[1] if len(sys.argv) > 1 else "cheetah-8_run"
env = higym.make(name)
obs, info = env.reset(seed=0)
env.action_space.seed(0)
episode_return, terminated, truncated, steps = 0.0, False, False, 0
while not (terminated or truncated):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    episode_return += reward
    steps += 1
print(f"{name}: {steps} steps, return {episode_return:.2f}, score {info['score']:.4f}, "
      f"obs dim {obs.shape[0]}, action dim {env.action_space.shape[0]}")
env.close()
