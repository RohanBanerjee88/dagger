"""The exchange rate between depths, and why the whole-track number is not one.

Stage B added an un-stratified whole-output SI-SDR (CLAUDE.md §7) to make "is
dilating net better?" decidable: the gain and its cost land in different depth
rows with no exchange rate between them. On real data (2026-08-23 verification
run) it came in at -13.17 dB while its own constituents read +46.91 (depth 1)
and +1.08 (depth 2) -- *below every depth it supposedly pools* in 271 of 288
rows, and empirically ANTI-correlated with depth 1 (r = -0.21).

The cause is not a wiring bug. SI-SDR fits ONE scalar over whatever samples it
is given. Score the whole track and the bit-exact solo copy pins that scalar
near 1, so a pure *level* error in the overlap region is charged at full price;
score each depth separately and the scalar floats per region and absorbs the
same error for free. ``TestTheWholeTrackNumberIsLevelDominated`` pins that
behaviour with a fixture where the estimate's SHAPE is held fixed and only its
overlap-region gain moves: every per-depth score is bit-identical across the
sweep while the whole-track number falls 13 dB.

So the whole-track number answers "how far off is the output, level included",
which is worth reporting, but it cannot weigh a depth-1 loss against a depth-2
gain -- the question it was added for. :func:`si_sdr_pooled_by_depth` is the
metric that can: it fits the scalar PER DEPTH (so a per-region level error is
discounted exactly as the per-depth tables already discount it) and then pools
the residual energies. Being a weighted mediant of the per-depth ratios, it is
provably bounded by them, which is what makes it a legitimate exchange rate and
is pinned below.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from dagger.metrics.sisdr import (
    depth_scale_factors,
    si_sdr,
    si_sdr_by_depth,
    si_sdr_pooled_by_depth,
)


def _two_region_case(overlap_gain: float, shape_error: float = 0.3):
    """A solo region copied bit-exactly, plus an overlap region with a fixed
    shape error and a variable level error.

    Returns ``(estimate, target, depth)``. Only ``overlap_gain`` moves between
    calls, so any metric that is scale-invariant per region must return the
    same value for every gain.
    """
    rng = np.random.default_rng(11)
    solo = rng.normal(size=3000)
    ovl = rng.normal(size=1000)
    err = rng.normal(size=1000)
    err -= err.dot(ovl) / ovl.dot(ovl) * ovl  # orthogonal, so it is pure shape
    err *= shape_error * np.linalg.norm(ovl) / np.linalg.norm(err)

    target = np.concatenate([solo, ovl])
    estimate = np.concatenate([solo, overlap_gain * (ovl + err)])
    depth = np.array([1] * 3000 + [2] * 1000)
    return estimate, target, depth


class TestTheWholeTrackNumberIsLevelDominated:
    """Characterization: this is the behaviour, and it is why a second metric
    is needed. Not a bug to fix -- a property to stop mis-reading."""

    def test_per_depth_scores_are_blind_to_the_level_error(self):
        baseline = si_sdr_by_depth(*_two_region_case(1.0))
        for gain in (3.0, 10.0, 100.0):
            swept = si_sdr_by_depth(*_two_region_case(gain))
            assert swept.keys() == baseline.keys()
            for k in baseline:
                assert swept[k] == pytest.approx(baseline[k], abs=1e-9), (
                    f"depth {k} moved with a pure level change -- si_sdr is "
                    "supposed to be scale-invariant"
                )

    def test_the_whole_track_number_collapses_on_the_same_fixture(self):
        values = [si_sdr(*_two_region_case(g)[:2]) for g in (1.0, 3.0, 10.0, 100.0)]
        assert values == sorted(values, reverse=True), values
        assert values[0] - values[-1] > 10.0, (
            f"expected a large whole-track penalty for a pure level error, got {values}"
        )

    def test_it_falls_below_every_depth_it_supposedly_pools(self):
        """The signature seen on real data in 271 of 288 rows."""
        estimate, target, depth = _two_region_case(10.0)
        whole = si_sdr(estimate, target)
        per_depth = si_sdr_by_depth(estimate, target, depth)
        assert whole < min(per_depth.values()), (
            f"whole {whole:.2f} vs depths {per_depth} -- fixture no longer "
            "reproduces the real-data signature"
        )


class TestPooledIsABoundedExchangeRate:
    def test_it_lies_between_the_best_and_worst_depth(self):
        for gain in (1.0, 3.0, 10.0, 100.0):
            estimate, target, depth = _two_region_case(gain)
            per_depth = si_sdr_by_depth(estimate, target, depth)
            pooled = si_sdr_pooled_by_depth(estimate, target, depth)
            assert min(per_depth.values()) - 1e-9 <= pooled <= max(per_depth.values()) + 1e-9, (
                f"pooled {pooled:.3f} outside [{min(per_depth.values()):.3f}, "
                f"{max(per_depth.values()):.3f}] at gain {gain}"
            )

    def test_it_is_invariant_to_a_per_region_level_error(self):
        """The whole point: it discounts level exactly as the per-depth tables
        already do, so a dilation comparison measures shape, not gain."""
        baseline = si_sdr_pooled_by_depth(*_two_region_case(1.0))
        for gain in (3.0, 10.0, 100.0):
            assert si_sdr_pooled_by_depth(*_two_region_case(gain)) == pytest.approx(
                baseline, abs=1e-9
            )

    def test_it_pools_rather_than_echoing_one_depth(self):
        estimate, target, depth = _two_region_case(1.0)
        per_depth = si_sdr_by_depth(estimate, target, depth)
        pooled = si_sdr_pooled_by_depth(estimate, target, depth)
        assert pooled != pytest.approx(min(per_depth.values()), abs=1e-6)
        assert pooled != pytest.approx(max(per_depth.values()), abs=1e-6)

    def test_a_perfect_region_stays_finite_when_another_region_is_not(self):
        """A bit-exact solo copy makes depth 1 ``+inf``. The pooled number must
        still be finite -- otherwise the overlap error is lost, which is the
        ``+-inf`` defect Phase 2 shipped, one level up."""
        target = np.concatenate([np.ones(100), np.array([1.0, 2.0, 3.0])])
        estimate = np.concatenate([np.ones(100), np.array([1.0, 2.0, 9.0])])
        depth = np.array([1] * 100 + [2] * 3)
        assert si_sdr_by_depth(estimate, target, depth)[1] == math.inf
        assert math.isfinite(si_sdr_pooled_by_depth(estimate, target, depth))

    def test_a_uniformly_better_estimate_pools_better(self):
        rng = np.random.default_rng(3)
        target = rng.normal(size=400)
        depth = np.array([1] * 200 + [2] * 200)
        good = target + 0.01 * rng.normal(size=400)
        bad = target + 0.20 * rng.normal(size=400)
        assert si_sdr_pooled_by_depth(good, target, depth) > si_sdr_pooled_by_depth(
            bad, target, depth
        )

    def test_depth_zero_is_excluded_like_the_stratified_metric(self):
        target = np.array([5.0, 5.0, 1.0, 1.0])
        estimate = np.array([0.0, 0.0, 1.0, 1.0])
        depth = np.array([0, 0, 1, 1])
        # Depth 0 is unscoreable, so a perfect depth-1 region pools to +inf
        # rather than being dragged down by samples nobody claimed.
        assert si_sdr_pooled_by_depth(estimate, target, depth) == math.inf

    def test_no_scoreable_region_is_nan(self):
        depth = np.zeros(4, dtype=int)
        assert math.isnan(si_sdr_pooled_by_depth(np.ones(4), np.ones(4), depth))

    def test_min_depth_restricts_the_pool(self):
        estimate, target, depth = _two_region_case(1.0)
        only_overlap = si_sdr_pooled_by_depth(estimate, target, depth, min_depth=2)
        assert only_overlap == pytest.approx(
            si_sdr_by_depth(estimate, target, depth)[2], abs=1e-9
        )


class TestLevelErrorIsMeasuredDirectly:
    """The quantity the whole-track number was implicitly charging for. Making
    it a column means the next run measures it instead of us inferring it from
    a discrepancy between two other metrics."""

    def test_it_recovers_a_known_overlap_gain(self):
        for gain in (2.0, 5.0, 20.0):
            estimate, target, depth = _two_region_case(gain, shape_error=0.0)
            scales = depth_scale_factors(estimate, target, depth)
            assert scales[1] == pytest.approx(1.0, rel=1e-6)
            assert scales[2] == pytest.approx(gain, rel=1e-6)

    def test_a_consistently_scaled_estimate_has_no_level_error(self):
        estimate, target, depth = _two_region_case(1.0)
        scales = depth_scale_factors(3.0 * estimate, target, depth)
        ratio = max(scales.values()) / min(scales.values())
        baseline = depth_scale_factors(estimate, target, depth)
        assert ratio == pytest.approx(
            max(baseline.values()) / min(baseline.values()), rel=1e-6
        ), "a global rescale is not a level ERROR -- only cross-region disagreement is"


class TestTheLevelErrorIsSigned:
    """`+` means the extractor is too LOUD in the deeper region, `-` too quiet.
    An unsigned max/min would collapse those onto one number, and they imply
    different bugs."""

    def _level(self, overlap_gain):
        from dagger.eval.systems import _level_error_db

        estimate, target, depth = _two_region_case(overlap_gain, shape_error=0.0)
        return _level_error_db(estimate, target, depth)

    def test_too_loud_is_positive(self):
        assert self._level(4.0) == pytest.approx(20.0 * math.log10(4.0), rel=1e-6)

    def test_too_quiet_is_negative(self):
        assert self._level(0.25) == pytest.approx(20.0 * math.log10(0.25), rel=1e-6)

    def test_a_consistent_level_is_zero(self):
        assert self._level(1.0) == pytest.approx(0.0, abs=1e-9)

    def test_one_scoreable_depth_is_nan(self):
        from dagger.eval.systems import _level_error_db

        target = np.array([1.0, 2.0, 3.0, 4.0])
        assert math.isnan(
            _level_error_db(target, target, np.ones(4, dtype=int))
        )
