"""The paper's gear-scaling ablations of Walker and Reacher (opt-in).

``import higym.ablations`` registers a Gymnasium id for each of the 24
variants the paper ran, e.g. ``higym/walker-256a25_run-v0``. Names follow
``ablation_scaling.py``: ``a<AA>`` scales gear, stiffness and damping by
``(baseline / n) ** (AA / 100)``; ``m`` conserves limb mass; ``p`` restores
DMControl's passive dynamics; ``s<SSS>`` sets Reacher's spring strength to
``SSS / 1000`` of max motor torque. The two ``s000`` variants (no spring)
are numerically unstable: the paper reports their runs stopping on
MuJoCo's ``mjWARN_BADQACC``.
"""

from higym import _register_ids

# Importing the module adds all 34 variants to dm_control's suites.
from higym.tasks import ablation_scaling  # noqa: F401

_WALKER_RUN = ("64a25", "64a50", "64a100", "256a25", "256a50", "256a100", "64a00m", "256a00m", "256a10m",
               "64a25m", "64a50m", "64a100m", "256a00p", "8a100")
# The reacher-easy variants exist in dm_control's suite but were never run.
_REACHER_HARD = ("64a25", "64a50", "64a100", "16a25", "16a50", "16a100", "64s000", "64s500", "64s1000", "4s000")

_PAIRS = [("walker", f"{v}_run") for v in _WALKER_RUN] + [("reacher", f"{v}_hard") for v in _REACHER_HARD]


def list_tasks() -> list[str]:
    """Names of the 24 ablation variants the paper ran, e.g. ``"walker-256a25_run"``."""
    return [f"{domain}-{task}" for domain, task in _PAIRS]


_register_ids(_PAIRS)
