"""Check an installed higym: all ids, one env per family, no training stack.

Run from outside the repo so the installed package, not src/, is imported.
"""

import sys

import higym

assert len(higym.list_tasks()) == 81, len(higym.list_tasks())
for name in ("quadruped-16_walk", "cheetah-8_run", "walker-8_walk", "finger-2_spin", "reacher-2_easy",
             "humanoid_CMU-80_stand"):
    env = higym.make(name)
    env.reset(seed=0)
    env.step(env.action_space.sample())
loaded = {"torch", "hydra", "wandb", "tensordict"} & set(sys.modules)
assert not loaded, loaded
print("ok", higym.__version__, higym.__file__)
