"""Phase 0: on-the-fly mixing (dagger.data.mixing)."""

from __future__ import annotations

import numpy as np
import pytest

from dagger.data.activity import segments_from_placement
from dagger.data.mixing import db_to_linear, mix_sources, stagger_offsets
from dagger.diarize.oracle import activity_matrix, solo_overlap_regions


class TestDbToLinear:
    def test_zero_db_is_unity_gain(self):
        assert db_to_linear(0.0) == pytest.approx(1.0)

    def test_plus_twenty_db_is_10x(self):
        assert db_to_linear(20.0) == pytest.approx(10.0)

    def test_minus_twenty_db_is_one_tenth(self):
        assert db_to_linear(-20.0) == pytest.approx(0.1)


class TestStaggerOffsetsLegacy:
    """``min_solo=0``: the original length-ratio-dependent chain staggering."""

    def test_first_offset_is_always_zero(self):
        offsets = stagger_offsets([1000, 2000, 3000], overlap=0.5)
        assert offsets[0] == 0

    def test_overlap_zero_is_back_to_back(self):
        offsets = stagger_offsets([1000, 2000], overlap=0.0)
        assert offsets == [0, 1000]

    def test_overlap_one_is_fully_overlapped(self):
        offsets = stagger_offsets([1000, 2000], overlap=1.0)
        assert offsets == [0, 0]

    def test_partial_overlap_stagger(self):
        offsets = stagger_offsets([1000, 2000], overlap=0.25)
        assert offsets == [0, 750]

    def test_offsets_are_deterministic(self):
        a = stagger_offsets([1000, 2000, 1500], overlap=0.5)
        b = stagger_offsets([1000, 2000, 1500], overlap=0.5)
        assert a == b

    def test_overlap_out_of_range_raises(self):
        with pytest.raises(ValueError):
            stagger_offsets([1000, 2000], overlap=1.5)
        with pytest.raises(ValueError):
            stagger_offsets([1000, 2000], overlap=-0.1)

    def test_negative_min_solo_raises(self):
        with pytest.raises(ValueError):
            stagger_offsets([1000, 2000], overlap=0.5, min_solo=-1)


class TestStaggerOffsetsMinSolo:
    """Regression test for the known issue: chain staggering starves 3+
    speakers of solo time depending on random length ratios (CLAUDE.md
    Phase 1 KNOWN ISSUE). ``min_solo`` must guarantee every speaker a
    contiguous solo window regardless of length ordering."""

    SAMPLE_RATE = 8000

    def _solo_run_lengths(self, lengths, overlap, min_solo):
        """Build the actual activity/solo masks from stagger_offsets' output
        and return each speaker's longest contiguous solo run (samples)."""
        offsets = stagger_offsets(lengths, overlap=overlap, min_solo=min_solo)
        speakers = [f"s{i}" for i in range(len(lengths))]
        segments = segments_from_placement(offsets, lengths, speakers, self.SAMPLE_RATE)
        n = max(offsets[i] + lengths[i] for i in range(len(lengths)))
        activity, speakers = activity_matrix(
            segments, num_samples=n, sample_rate=self.SAMPLE_RATE, speakers=speakers
        )
        solo, _ = solo_overlap_regions(activity)

        longest = []
        for row in solo:
            runs, cur = [], 0
            for v in row:
                if v:
                    cur += 1
                else:
                    if cur:
                        runs.append(cur)
                    cur = 0
            if cur:
                runs.append(cur)
            longest.append(max(runs) if runs else 0)
        return longest

    def test_middle_speaker_starved_without_min_solo(self):
        """The documented failure mode: L2 < L1 means the legacy (min_solo=0)
        chain leaves the middle speaker with no solo time at all."""
        lengths = [4000, 1000, 4000]  # L2 < L1: middle speaker starved
        longest = self._solo_run_lengths(lengths, overlap=0.5, min_solo=0)
        assert longest[1] == 0

    def test_min_solo_guarantees_every_speaker_a_solo_window(self):
        lengths = [4000, 1000, 4000]  # same starving case as above
        min_solo = 300
        longest = self._solo_run_lengths(lengths, overlap=0.5, min_solo=min_solo)
        for i, run in enumerate(longest):
            expected = min(min_solo, lengths[i])
            assert run >= expected, f"speaker {i} solo run {run} < guaranteed {expected}"

    def test_min_solo_guarantee_holds_across_random_length_ratios(self):
        rng = np.random.default_rng(0)
        min_solo = 200
        for _ in range(20):
            lengths = [int(x) for x in rng.integers(500, 5000, size=3)]
            longest = self._solo_run_lengths(lengths, overlap=0.5, min_solo=min_solo)
            for i, run in enumerate(longest):
                expected = min(min_solo, lengths[i])
                assert run >= expected


class TestMixSources:
    def test_default_gains_and_offsets_sum_to_mixture(self):
        sources = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0])]
        padded, mixture = mix_sources(sources)
        np.testing.assert_allclose(mixture, padded.sum(axis=0))
        np.testing.assert_allclose(mixture, [2.0, 3.0, 4.0])

    def test_gains_scale_each_source(self):
        sources = [np.array([1.0, 1.0]), np.array([1.0, 1.0])]
        padded, mixture = mix_sources(sources, gains=[2.0, 0.5])
        np.testing.assert_allclose(padded[0], [2.0, 2.0])
        np.testing.assert_allclose(padded[1], [0.5, 0.5])
        np.testing.assert_allclose(mixture, [2.5, 2.5])

    def test_offsets_place_sources(self):
        sources = [np.array([1.0, 1.0]), np.array([2.0, 2.0])]
        padded, mixture = mix_sources(sources, offsets=[0, 2], length_mode="max")
        assert padded.shape == (2, 4)
        np.testing.assert_allclose(padded[0], [1.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(padded[1], [0.0, 0.0, 2.0, 2.0])
        np.testing.assert_allclose(mixture, [1.0, 1.0, 2.0, 2.0])

    def test_length_mode_max_keeps_solo_tail(self):
        sources = [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0])]
        padded, mixture = mix_sources(sources, offsets=[0, 0], length_mode="max")
        assert padded.shape == (2, 3)

    def test_length_mode_min_truncates(self):
        sources = [np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0])]
        padded, mixture = mix_sources(sources, offsets=[0, 0], length_mode="min")
        assert padded.shape == (2, 2)

    def test_explicit_length_overrides_length_mode(self):
        sources = [np.array([1.0, 1.0, 1.0])]
        padded, mixture = mix_sources(sources, length=5)
        assert padded.shape == (1, 5)
        np.testing.assert_allclose(padded[0], [1.0, 1.0, 1.0, 0.0, 0.0])

    def test_no_sources_raises(self):
        with pytest.raises(ValueError):
            mix_sources([])

    def test_mismatched_gains_length_raises(self):
        with pytest.raises(ValueError):
            mix_sources([np.array([1.0]), np.array([1.0])], gains=[1.0])

    def test_unknown_length_mode_raises(self):
        with pytest.raises(ValueError):
            mix_sources([np.array([1.0])], length_mode="bogus")

    def test_source_clipped_when_shorter_output_length_given(self):
        sources = [np.array([1.0, 2.0, 3.0, 4.0])]
        padded, mixture = mix_sources(sources, length=2)
        np.testing.assert_allclose(padded[0], [1.0, 2.0])
