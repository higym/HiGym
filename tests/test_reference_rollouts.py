"""Short seeded rollouts, one task per family, against references recorded
under the paper's software stack (python 3.10, numpy 1.24.4, gymnasium 0.29.1)
with scripts/record_reference_rollouts.py.

References are kept per platform: MuJoCo is not bitwise reproducible across
CPU platforms, and a contact-rich model like humanoid_CMU-80 ends up on a
different trajectory within 50 steps. On one platform they reproduce
bit-for-bit across python 3.10/3.12, numpy 1.24/1.26 and gymnasium 0.29/1.x.
"""

import pathlib

import numpy as np
import pytest
from helpers import REFERENCE_TASKS, platform_tag, reference_rollout

REFERENCE_FILE = pathlib.Path(__file__).parent / "data" / "reference_rollouts" / f"{platform_tag()}.npz"

pytestmark = pytest.mark.skipif(
    not REFERENCE_FILE.exists(), reason=f"no reference rollouts recorded for {platform_tag()}"
)


@pytest.mark.parametrize("name", REFERENCE_TASKS)
def test_rollout_matches_reference(name):
    reference = np.load(REFERENCE_FILE)
    obs, rewards = reference_rollout(name)
    np.testing.assert_allclose(obs, reference[f"{name}.obs"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(rewards, reference[f"{name}.reward"], rtol=1e-6, atol=1e-8)
