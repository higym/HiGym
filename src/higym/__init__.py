"""HiGym: high-DoF continuous-control environments built on dm_control."""

import gymnasium

from higym._suite import load, table1_tasks
from higym.env import HiGymEnv

__version__ = "0.1.0"
__all__ = ["HiGymEnv", "list_tasks", "load", "make"]


def list_tasks() -> list[str]:
    """Names of the paper's 81 Table 1 tasks, e.g. ``"cheetah-128_run"``."""
    return [f"{domain}-{task}" for domain, task in table1_tasks()]


def make(name: str, **kwargs) -> gymnasium.Env:
    """Shorthand for ``gymnasium.make(f"higym/{name}-v0", **kwargs)``."""
    return gymnasium.make(f"higym/{name}-v0", **kwargs)


def _register_ids(pairs):
    for domain, task in pairs:
        gymnasium.register(
            id=f"higym/{domain}-{task}-v0",
            entry_point="higym.env:HiGymEnv",
            kwargs={"domain": domain, "task": task},
        )


_register_ids(table1_tasks())
