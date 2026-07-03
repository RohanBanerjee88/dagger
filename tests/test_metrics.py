"""SI-SDR sanity checks."""

import numpy as np

from dagger.metrics.sisdr import si_sdr, si_sdr_regionwise


def test_perfect_estimate_is_inf():
    target = np.sin(np.linspace(0, 10, 200))
    assert si_sdr(target, target) == float("inf")


def test_scale_invariance():
    target = np.random.default_rng(0).standard_normal(500)
    # a pure gain change must not affect SI-SDR (still perfect)
    assert si_sdr(3.7 * target, target) == float("inf")


def test_silent_target_is_nan():
    assert np.isnan(si_sdr(np.ones(10), np.zeros(10)))


def test_added_noise_lowers_score():
    rng = np.random.default_rng(1)
    target = rng.standard_normal(1000)
    noisy = target + 0.1 * rng.standard_normal(1000)
    score = si_sdr(noisy, target)
    assert np.isfinite(score) and score > 10  # small noise -> high but finite SI-SDR


def test_regionwise_masks():
    target = np.arange(10, dtype=float)
    est = target.copy()
    mask = np.zeros(10, dtype=bool)
    mask[2:6] = True
    assert si_sdr_regionwise(est, target, mask) == float("inf")
    assert np.isnan(si_sdr_regionwise(est, target, np.zeros(10, dtype=bool)))
