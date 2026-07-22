"""The V_i enrollment-variance pre-check (CLAUDE.md §2, §5 Phase 2).

"The gate can't check its own enrollment": if a speaker's enrollment is itself
contaminated (e.g. a mislabeled overlap frame was mistaken for solo audio), the
resulting mean embedding is unreliable, and a *margin* computed against it would
happily "pass" for the wrong reason. This check runs before the margin gate and
rejects an enrollment whose per-dimension variance across enrollment clips
(:attr:`dagger.enroll.topk.EnrollmentResult.variance`) is too high to trust.
"""

from __future__ import annotations

import numpy as np


def enrollment_variance_ok(variance: np.ndarray, max_mean_variance: float) -> bool:
    """``True`` iff the enrollment's mean per-dimension variance is within bounds.

    ``variance`` is ``V_i`` (:attr:`~dagger.enroll.topk.EnrollmentResult.variance`),
    computed across a speaker's top-K solo clips. A high variance means the
    enrollment clips disagree with each other -- plausibly because one of them
    wasn't actually solo -- so the margin gate downstream should not be trusted
    for this speaker until enrollment is fixed (or the speaker is dropped).
    """
    return float(np.mean(np.asarray(variance, dtype=np.float64))) <= max_mean_variance
