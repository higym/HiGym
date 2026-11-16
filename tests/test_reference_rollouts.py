"""Short seeded rollouts, one task per family, against references recorded
under the paper's software stack (python 3.10, numpy 1.24.4, gymnasium 0.29.1).

Compared with a tight tolerance rather than bit-for-bit, because MuJoCo is
not bitwise reproducible across CPU platforms.
"""

import pathlib

import numpy as np
import pytest

from helpers import reference_rollout

REFERENCE = np.load(pathlib.Path(__file__).parent / "data" / "reference_rollouts.npz")
NAMES = sorted({key.rsplit(".", 1)[0] for key in REFERENCE.files})


@pytest.mark.parametrize("name", NAMES)
def test_rollout_matches_reference(name):
    obs, rewards = reference_rollout(name)
    np.testing.assert_allclose(obs, REFERENCE[f"{name}.obs"], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(rewards, REFERENCE[f"{name}.reward"], rtol=1e-6, atol=1e-8)
