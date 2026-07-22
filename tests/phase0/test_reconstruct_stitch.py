"""Phase 0: soft-mask stitching / reconstruction (dagger.reconstruct.stitch).

Covers the two settled facts from CLAUDE.md §2 this module encodes: solo
regions are copied bit-exact (never run through the extractor), and the
solo/overlap windows form a partition of unity with a smooth crossfade.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.audio.provenance import ResidualInAudioPathError, TrackedSignal, original_mixture
from dagger.extract.base import NullExtractor
from dagger.reconstruct.stitch import crossfade_windows, reconstruct_all, reconstruct_speaker


class TestCrossfadeWindows:
    def test_hard_masks_when_fade_is_zero(self):
        solo_i = np.array([1.0, 1.0, 0.0, 0.0])
        activity_i = np.array([1.0, 1.0, 1.0, 1.0])
        w_e, w_o = crossfade_windows(solo_i, activity_i, fade=0)
        np.testing.assert_array_equal(w_e, solo_i)
        np.testing.assert_array_equal(w_o, [0.0, 0.0, 1.0, 1.0])

    def test_partition_of_unity_holds_hard(self):
        solo_i = np.array([1.0, 0.0, 1.0, 0.0])
        activity_i = np.array([1.0, 1.0, 1.0, 0.0])
        w_e, w_o = crossfade_windows(solo_i, activity_i, fade=0)
        np.testing.assert_allclose(w_e + w_o, activity_i)

    def test_partition_of_unity_holds_with_fade(self):
        rng = np.random.default_rng(0)
        n = 200
        activity_i = (rng.random(n) > 0.2).astype(np.float64)
        solo_i = activity_i * (rng.random(n) > 0.5).astype(np.float64)
        w_e, w_o = crossfade_windows(solo_i, activity_i, fade=5)
        np.testing.assert_allclose(w_e + w_o, activity_i, atol=1e-10)

    def test_windows_are_zero_outside_activity(self):
        # fade's box kernel is (2*fade + 1) samples; keep the array at least
        # that long (crossfade_windows only supports fade <= len(activity_i) // 2).
        solo_i = np.zeros(8)
        activity_i = np.zeros(8)
        w_e, w_o = crossfade_windows(solo_i, activity_i, fade=3)
        np.testing.assert_array_equal(w_e, np.zeros(8))
        np.testing.assert_array_equal(w_o, np.zeros(8))

    def test_fade_ramp_stays_within_unit_range(self):
        solo_i = np.array([1.0] * 20 + [0.0] * 20)
        activity_i = np.ones(40)
        w_e, w_o = crossfade_windows(solo_i, activity_i, fade=5)
        assert (w_e >= 0.0).all() and (w_e <= 1.0).all()
        assert (w_o >= 0.0).all() and (w_o <= 1.0).all()


class TestReconstructSpeaker:
    def test_solo_regions_copied_bit_exact_with_null_extractor(self):
        """Phase 0 DoD: 'copying solo regions recovers solo audio exactly.'"""
        x = original_mixture(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        overlap = np.array([0.0, 1.0, 1.0, 0.0, 0.0])
        x_O = x.masked(overlap, label="x_O")
        activity_i = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        solo_i = np.array([1.0, 0.0, 0.0, 1.0, 1.0])

        out = reconstruct_speaker(
            x, x_O, activity_i, solo_i, embedding=np.zeros(1), extractor=NullExtractor()
        )
        # NullExtractor outputs silence on overlap, so with fade=0 the result
        # is exactly the mixture on solo frames and exactly zero elsewhere.
        np.testing.assert_array_equal(out, [1.0, 0.0, 0.0, 4.0, 5.0])

    def test_extraction_only_applies_on_overlap_weighted_region(self):
        class _ConstantExtractor:
            def extract(self, x_O, embedding):
                return np.full_like(np.asarray(x_O), 9.0)

        x = original_mixture(np.array([1.0, 2.0, 3.0, 4.0]))
        overlap = np.array([0.0, 1.0, 1.0, 0.0])
        x_O = x.masked(overlap, label="x_O")
        activity_i = np.array([1.0, 1.0, 1.0, 1.0])
        solo_i = np.array([1.0, 0.0, 0.0, 1.0])

        out = reconstruct_speaker(
            x, x_O, activity_i, solo_i, embedding=np.zeros(1), extractor=_ConstantExtractor()
        )
        np.testing.assert_array_equal(out, [1.0, 9.0, 9.0, 4.0])

    def test_raises_if_extractor_is_handed_a_residual(self):
        x = original_mixture(np.array([1.0, 2.0, 3.0]))
        residual = x - TrackedSignal(np.array([0.1, 0.1, 0.1]))
        activity_i = np.array([1.0, 1.0, 1.0])
        solo_i = np.array([0.0, 0.0, 0.0])
        with pytest.raises(ResidualInAudioPathError):
            reconstruct_speaker(
                x, residual, activity_i, solo_i, embedding=np.zeros(1), extractor=NullExtractor()
            )


class TestReconstructAll:
    def test_shape_matches_activity(self):
        x = original_mixture(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
        overlap = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        x_O = x.masked(overlap)
        activity = np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        solo = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        out = reconstruct_all(x, x_O, activity, solo, embeddings=None, extractor=NullExtractor())
        assert out.shape == activity.shape

    def test_works_with_embeddings_none_phase0_case(self):
        x = original_mixture(np.ones(4))
        x_O = x.masked(np.array([0.0, 1.0, 1.0, 0.0]))
        activity = np.ones((1, 4))
        solo = np.array([[1.0, 0.0, 0.0, 1.0]])
        out = reconstruct_all(x, x_O, activity, solo, embeddings=None, extractor=NullExtractor())
        np.testing.assert_array_equal(out[0], [1.0, 0.0, 0.0, 1.0])
