"""The HiGym tasks, as registered on dm_control's domain suites."""

import importlib
import re

from dm_control import suite

# Importing the task modules adds their tasks to dm_control's domain suites.
from higym.tasks import cheetah, finger, humanoid_CMU, quadruped, reacher, walker  # noqa: F401

DOMAINS = ("quadruped", "cheetah", "walker", "finger", "reacher", "humanoid_CMU")

# DoF sizes in the paper's Table 1. The task modules register a few more
# (cheetah 4 and 512, walker 4, humanoid_CMU 56), which get no Gymnasium id.
TABLE1_DOF = {
    "quadruped": (16, 32, 64, 128, 256, 512),
    "cheetah": (8, 16, 32, 64, 128, 256),
    "walker": (8, 16, 32, 64, 128, 256),
    "finger": (2, 4, 8, 16, 32, 64),
    "reacher": (2, 4, 8, 16, 32, 64),
    "humanoid_CMU": (80,),
}

_SIZED = re.compile(r"^(\d+)_")


def custom_tasks(domain: str) -> list[str]:
    """Names of the HiGym tasks registered on ``domain``, in registration order."""
    return list(importlib.import_module(f"dm_control.suite.{domain}").SUITE.tagged("custom"))


def table1_tasks() -> list[tuple[str, str]]:
    """``(domain, task)`` for the Table 1 tasks, ordered by domain, DoF, then task."""
    pairs = []
    for domain in DOMAINS:
        sized = []
        for task in custom_tasks(domain):
            match = _SIZED.match(task)
            if match and int(match.group(1)) in TABLE1_DOF[domain]:
                sized.append((int(match.group(1)), task))
        pairs += [(domain, task) for _, task in sorted(sized, key=lambda s: s[0])]
    return pairs


def load(domain: str, task: str, seed: int | None = None):
    """The dm_control environment for a HiGym task, built as ``suite.load`` builds it.

    ``seed`` seeds the task's random state (initial states, targets, terrain).
    """
    if domain not in DOMAINS or task not in custom_tasks(domain):
        raise ValueError(
            f"{domain}-{task} is not a HiGym task; "
            "dm_control's own tasks load with dm_control.suite.load"
        )
    return suite.load(domain, task, task_kwargs={"random": seed}, visualize_reward=False)
