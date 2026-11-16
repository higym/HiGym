"""Record the rollouts tests/test_reference_rollouts.py compares against.

    python scripts/record_reference_rollouts.py [out.npz]

Writes tests/data/reference_rollouts/<platform>.npz unless a path is given.
Record under the paper's stack (pip install -e ".[paper]") on the platform
the file is for.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np  # noqa: E402
from helpers import REFERENCE_TASKS, platform_tag, reference_rollout  # noqa: E402

out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "tests" / "data" / "reference_rollouts" / f"{platform_tag()}.npz")
arrays = {}
for name in REFERENCE_TASKS:
    arrays[f"{name}.obs"], arrays[f"{name}.reward"] = reference_rollout(name)
np.savez_compressed(out, **arrays)
print(f"wrote {out} ({platform_tag()})")
