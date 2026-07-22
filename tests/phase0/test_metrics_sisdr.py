"""Phase 0: SI-SDR metrics (dagger.metrics.sisdr)."""

from __future__ import annotations

import numpy as np
import pytest

from dagger.metrics.sisdr import si_sdr, si_sdr_best_permutation, si_sdr_regionwise


class TestSiSdr:
    def test_identical_signals_give_inf(self):
        x = np.array([1.0, -2.0, 3.0, 0.5])
        assert si_sdr(x, x) == float("inf")

    def test_scaled_copy_is_still_perfect_scale_invariance(self):
        target = np.array([1.0, -2.0, 3.0, 0.5])
        estimate = target * 2.5
        assert si_sdr(estimate, target) == float("inf")

    def test_silent_estimate_against_real_target_is_negative_inf(self):
        target = np.array([1.0, -2.0, 3.0, 0.5])
        estimate = np.zeros_like(target)
        assert si_sdr(estimate, target) == float("-inf")

    def test_silent_target_is_nan(self):
        target = np.zeros(4)
        estimate = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.isnan(si_sdr(estimate, target))

    def test_orthogonal_noise_matches_manual_formula(self):
        target = np.array([1.0, 0.0, 0.0, 0.0])
        noise = np.array([0.0, 1.0, 0.0, 0.0])
        estimate = target + 0.5 * noise
        expected = 10.0 * np.log10(1.0 / (0.5 ** 2))
        assert si_sdr(estimate, target) == pytest.approx(expected, rel=1e-6)

    def test_worse_than_silence_is_negative(self):
        target = np.array([1.0, 0.0, 0.0, 0.0])
        estimate = np.array([0.0, 5.0, 0.0, 0.0])  # all noise, no signal component
        assert si_sdr(estimate, target) < 0


class TestSiSdrRegionwise:
    def test_matches_manual_masked_si_sdr(self):
        target = np.array([1.0, 2.0, 3.0, 4.0])
        estimate = np.array([1.0, 2.0, 30.0, 40.0])  # garbage outside the mask
        mask = np.array([True, True, False, False])
        expected = si_sdr(estimate[mask.astype(bool)], target[mask.astype(bool)])
        assert si_sdr_regionwise(estimate, target, mask) == pytest.approx(expected)

    def test_all_false_mask_is_nan(self):
        target = np.array([1.0, 2.0])
        estimate = np.array([1.0, 2.0])
        mask = np.array([False, False])
        assert np.isnan(si_sdr_regionwise(estimate, target, mask))

    def test_accepts_float_mask(self):
        target = np.array([1.0, 2.0, 3.0])
        estimate = np.array([1.0, 2.0, 3.0])
        mask = np.array([1.0, 1.0, 0.0])
        assert si_sdr_regionwise(estimate, target, mask) == float("inf")


class TestSiSdrBestPermutation:
    def test_recovers_shuffled_permutation(self):
        targets = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        # estimates are the targets swapped -> best perm should un-swap them
        estimates = targets[[1, 0]]
        scores, perm = si_sdr_best_permutation(estimates, targets)
        assert perm == (1, 0)
        assert all(s == float("inf") for s in scores)

    def test_identity_permutation_when_already_aligned(self):
        targets = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        scores, perm = si_sdr_best_permutation(targets.copy(), targets)
        assert perm == (0, 1, 2)

    def test_matches_direct_si_sdr_at_best_permutation(self):
        rng = np.random.default_rng(0)
        targets = rng.normal(size=(2, 16))
        estimates = rng.normal(size=(2, 16))
        scores, perm = si_sdr_best_permutation(estimates, targets)
        for i in range(2):
            assert scores[i] == pytest.approx(si_sdr(estimates[perm[i]], targets[i]))
