"""Tests for ``enroll.budget_ms`` -- the enrollment-audio cap (Phase 2).

Coarse-to-fine blends the solo-derived embedding toward one computed from
extracted overlap audio, which only pays when the solo clip is the weaker
estimate. At the default ~1 s of clean solo it isn't, and refinement costs
0.2-1.1 dB. ``budget_ms`` starves enrollment deliberately so the crossover can
be located (or ruled out).

The properties that make the sweep interpretable, and that these tests pin:

* it truncates audio and changes nothing else -- crucially not the clip COUNT,
  since that would move ``V_i`` and confound the experiment;
* it does not veto a speaker when the budget falls below ``min_clip_ms``
  (starving is the point);
* omitting it reproduces the pre-existing behavior exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.enroll.topk import enroll_speaker, select_topk_solo_clips

SAMPLE_RATE = 8000


def _mixture(num_samples: int) -> np.ndarray:
    t = np.arange(num_samples, dtype=np.float64) / SAMPLE_RATE
    return np.sin(2.0 * np.pi * 220.0 * t)


def _one_run_scene(solo_ms: float, total_ms: float = 4000.0):
    """One speaker with a single contiguous solo run starting at sample 0."""
    total = int(total_ms / 1000.0 * SAMPLE_RATE)
    solo_len = int(solo_ms / 1000.0 * SAMPLE_RATE)
    solo = np.zeros(total, dtype=np.float64)
    solo[:solo_len] = 1.0
    activity = np.ones(total, dtype=np.float64)
    return _mixture(total), solo, activity


class TestBudgetTruncation:
    def test_default_is_none_and_keeps_the_whole_clip(self):
        mixture, solo, _ = _one_run_scene(solo_ms=1000.0)

        clips = select_topk_solo_clips(mixture, solo, SAMPLE_RATE, min_clip_ms=500.0)

        assert len(clips) == 1
        assert clips[0].size == SAMPLE_RATE  # the full 1000 ms

    def test_budget_truncates_to_the_requested_duration(self):
        mixture, solo, _ = _one_run_scene(solo_ms=1000.0)

        clips = select_topk_solo_clips(
            mixture, solo, SAMPLE_RATE, min_clip_ms=500.0, budget_ms=300.0
        )

        assert clips[0].size == int(0.3 * SAMPLE_RATE)
        # Taken from the clip's start, deterministically -- no RNG anywhere, so
        # a sweep point is exactly reproducible.
        np.testing.assert_array_equal(clips[0], mixture[: int(0.3 * SAMPLE_RATE)])

    def test_a_budget_larger_than_the_clip_is_a_no_op(self):
        mixture, solo, _ = _one_run_scene(solo_ms=600.0)

        clips = select_topk_solo_clips(
            mixture, solo, SAMPLE_RATE, min_clip_ms=500.0, budget_ms=5000.0
        )

        assert clips[0].size == int(0.6 * SAMPLE_RATE)

    def test_budget_below_min_clip_ms_still_yields_a_clip(self):
        """`min_clip_ms` filters which runs qualify, using their ORIGINAL length;
        the budget then truncates what qualified. If the order were reversed, a
        budget under the stability floor would reject every speaker and the
        starved end of the sweep would be unrunnable -- which is exactly the end
        the experiment cares about."""
        mixture, solo, _ = _one_run_scene(solo_ms=1000.0)

        clips = select_topk_solo_clips(
            mixture, solo, SAMPLE_RATE, min_clip_ms=500.0, budget_ms=150.0
        )

        assert len(clips) == 1
        assert clips[0].size == int(0.15 * SAMPLE_RATE)

    def test_a_run_shorter_than_min_clip_ms_is_still_rejected(self):
        """The budget must not accidentally rescue a run that was too short to
        begin with -- that would change which speakers are enrollable and break
        pairing across sweep points."""
        mixture, solo, _ = _one_run_scene(solo_ms=300.0)

        clips = select_topk_solo_clips(
            mixture, solo, SAMPLE_RATE, min_clip_ms=500.0, budget_ms=150.0
        )

        assert clips == []


class TestBudgetDoesNotChangeClipCount:
    def _three_run_scene(self):
        total = 6 * SAMPLE_RATE
        solo = np.zeros(total, dtype=np.float64)
        for start_s, length_s in ((0, 1.5), (2, 1.0), (4, 0.8)):
            start = int(start_s * SAMPLE_RATE)
            solo[start : start + int(length_s * SAMPLE_RATE)] = 1.0
        return _mixture(total), solo, np.ones(total, dtype=np.float64)

    @pytest.mark.parametrize("budget_ms", [None, 700.0, 300.0, 100.0])
    def test_clip_count_is_independent_of_the_budget(self, budget_ms):
        """Capping a TOTAL budget across clips would drop clips entirely, which
        would collapse ``V_i`` (the variance across clips) toward zero and
        confound the sweep with a change in the variance signal. Capping per
        clip keeps the count fixed."""
        mixture, solo, _ = self._three_run_scene()

        clips = select_topk_solo_clips(
            mixture, solo, SAMPLE_RATE, k=3, min_clip_ms=500.0, budget_ms=budget_ms
        )

        assert len(clips) == 3

    def test_clips_stay_ordered_longest_first_after_truncation(self):
        mixture, solo, _ = self._three_run_scene()

        clips = select_topk_solo_clips(
            mixture, solo, SAMPLE_RATE, k=3, min_clip_ms=500.0, budget_ms=1200.0
        )

        # Only the 1.5 s run exceeds the budget, so it is capped; selection order
        # is by original length, which the truncation must not reshuffle.
        sizes = [clip.size for clip in clips]
        assert sizes == [int(1.2 * SAMPLE_RATE), int(1.0 * SAMPLE_RATE), int(0.8 * SAMPLE_RATE)]


class TestEnrollSpeakerPassesItThrough:
    class _LengthEncoder:
        """Embeds a clip as its length, so the enrollment embedding reveals
        exactly how much audio reached the encoder."""

        def embed(self, waveform, sample_rate):
            return np.array([float(np.asarray(waveform).size)], dtype=np.float64)

    def test_budget_reaches_the_encoder(self):
        mixture, solo, activity = _one_run_scene(solo_ms=1000.0)

        full = enroll_speaker(
            mixture, solo, activity, SAMPLE_RATE, self._LengthEncoder(), min_clip_ms=500.0
        )
        starved = enroll_speaker(
            mixture, solo, activity, SAMPLE_RATE, self._LengthEncoder(),
            min_clip_ms=500.0, budget_ms=250.0,
        )

        assert full.embedding[0] == SAMPLE_RATE
        assert starved.embedding[0] == int(0.25 * SAMPLE_RATE)
        # Same number of clips either way, so V_i keeps its meaning.
        assert full.clip_count == starved.clip_count == 1
