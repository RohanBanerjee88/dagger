"""Phase 1: top-K solo-clip enrollment (dagger.enroll.topk).

Enrollment must only ever draw from a speaker's solo region -- CLAUDE.md's
named Phase 1 red flag is "enrollment taken from overlap by accident", which
this module guards against with an assertion that must fail loudly (a plain
``ValueError``, never silently caught) rather than the benign
``NoSoloRegionError`` skip path.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.enroll.topk import (
    EnrollmentResult,
    NoSoloRegionError,
    enroll_speaker,
    mean_embedding,
    select_topk_solo_clips,
)

SAMPLE_RATE = 1000


class TestSelectTopkSoloClips:
    def test_returns_longest_runs_first(self):
        mixture = np.arange(20, dtype=np.float64)
        solo_i = np.zeros(20)
        solo_i[0:3] = 1.0   # run of 3
        solo_i[5:15] = 1.0  # run of 10
        solo_i[17:19] = 1.0  # run of 2
        clips = select_topk_solo_clips(mixture, solo_i, SAMPLE_RATE, k=3, min_clip_ms=0.0)
        assert [len(c) for c in clips] == [10, 3, 2]
        np.testing.assert_array_equal(clips[0], mixture[5:15])

    def test_k_limits_number_of_clips(self):
        mixture = np.arange(20, dtype=np.float64)
        solo_i = np.zeros(20)
        solo_i[0:5] = 1.0
        solo_i[10:15] = 1.0
        clips = select_topk_solo_clips(mixture, solo_i, SAMPLE_RATE, k=1, min_clip_ms=0.0)
        assert len(clips) == 1

    def test_runs_shorter_than_min_clip_ms_are_dropped(self):
        mixture = np.arange(20, dtype=np.float64)
        solo_i = np.zeros(20)
        solo_i[0:2] = 1.0  # 2 samples = 2ms @ 1000Hz
        solo_i[10:18] = 1.0  # 8 samples = 8ms
        clips = select_topk_solo_clips(mixture, solo_i, SAMPLE_RATE, k=5, min_clip_ms=5.0)
        assert len(clips) == 1
        np.testing.assert_array_equal(clips[0], mixture[10:18])

    def test_no_solo_region_raises_no_solo_region_error(self):
        mixture = np.arange(10, dtype=np.float64)
        solo_i = np.zeros(10)
        with pytest.raises(NoSoloRegionError):
            select_topk_solo_clips(mixture, solo_i, SAMPLE_RATE, min_clip_ms=0.0)

    def test_overlap_contamination_is_a_plain_value_error_not_no_solo(self):
        """The overlap-contamination guard must be a distinct, non-catchable-
        by-skip-callers error type from the benign no-solo-region case."""
        mixture = np.arange(10, dtype=np.float64)
        solo_i = np.array([1.0] * 10)  # claims solo everywhere...
        activity_i = np.array([0.0] * 5 + [1.0] * 5)  # ...but only active half the time
        with pytest.raises(ValueError) as exc_info:
            select_topk_solo_clips(mixture, solo_i, SAMPLE_RATE, activity_i=activity_i, min_clip_ms=0.0)
        assert not isinstance(exc_info.value, NoSoloRegionError)

    def test_solo_subset_of_activity_passes_the_guard(self):
        mixture = np.arange(10, dtype=np.float64)
        solo_i = np.array([1.0] * 5 + [0.0] * 5)
        activity_i = np.array([1.0] * 10)
        clips = select_topk_solo_clips(mixture, solo_i, SAMPLE_RATE, activity_i=activity_i, min_clip_ms=0.0)
        assert len(clips) == 1


class TestMeanEmbedding:
    def test_mean_and_variance_across_clips(self, fake_encoder):
        clips = [np.ones(10) * 1.0, np.ones(10) * 3.0]
        mean, var = mean_embedding(clips, SAMPLE_RATE, fake_encoder)
        embeddings = np.stack([fake_encoder.embed(c, SAMPLE_RATE) for c in clips])
        np.testing.assert_allclose(mean, embeddings.mean(axis=0))
        np.testing.assert_allclose(var, embeddings.var(axis=0))

    def test_empty_clips_raises_no_solo_region_error(self, fake_encoder):
        with pytest.raises(NoSoloRegionError):
            mean_embedding([], SAMPLE_RATE, fake_encoder)


class TestEnrollSpeaker:
    def test_builds_enrollment_result(self, fake_encoder):
        mixture = np.linspace(-1.0, 1.0, 100)
        solo_i = np.zeros(100)
        solo_i[10:60] = 1.0
        activity_i = solo_i.copy()
        result = enroll_speaker(
            mixture, solo_i, activity_i, SAMPLE_RATE, fake_encoder, k=3, min_clip_ms=0.0
        )
        assert isinstance(result, EnrollmentResult)
        assert result.clip_count == 1
        assert result.embedding.shape == (3,)
        assert result.variance.shape == (3,)

    def test_no_solo_region_propagates(self, fake_encoder):
        mixture = np.zeros(50)
        solo_i = np.zeros(50)
        activity_i = np.zeros(50)
        with pytest.raises(NoSoloRegionError):
            enroll_speaker(mixture, solo_i, activity_i, SAMPLE_RATE, fake_encoder, min_clip_ms=0.0)
