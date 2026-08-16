"""Refinement must survive a too-short overlap run, and say so distinctly.

Under oracle diarization the scene scheduler gives every speaker a long overlap
tail, so this path never ran. A real diarizer's boundaries are jittery and an
overlap-only run of a few samples is routine — TitaNet then receives fewer
samples than one mel frame and NeMo raises, which killed a Phase 3 run at scene
9 of 150.

The second class below guards the subtler hazard the fix introduced: a skip is
recorded as `accepted=False`, which is indistinguishable from a *rejection* to
any assertion that only checks `accepted`. That is precisely how one existing
Phase 2 test started passing vacuously when this landed.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.audio.provenance import original_mixture
from dagger.extract.base import Extractor
from dagger.refine.coarse_to_fine import refine_embeddings

SAMPLE_RATE = 8000

GATE = dict(tau_margin=-10.0, max_mean_variance=10.0,
            min_vad_coverage=0.0, max_artifact_score=100.0)


class _PassthroughExtractor(Extractor):
    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        return x_O * 0.5


class _ExplodingEncoder:
    """Stands in for TitaNet's real constraint: too-short input raises.

    NeMo's message is "normalize_batch with `per_feature` normalize_type
    received a tensor of length 1", i.e. fewer samples than one mel frame.
    """

    MIN_SAMPLES = 400  # ~50 ms at 8 kHz

    def __init__(self):
        self.calls = 0

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        self.calls += 1
        if np.asarray(waveform).shape[0] < self.MIN_SAMPLES:
            raise ValueError(
                "normalize_batch with `per_feature` normalize_type received a "
                "tensor of length 1."
            )
        w = np.asarray(waveform, dtype=np.float64)
        return np.array([float(w.mean()), float(np.sqrt(np.mean(w**2))), 0.0])


def _scene_with_overlap_run(run_samples: int, total: int = 4000):
    """Two speakers whose overlap-only region is exactly ``run_samples`` long."""
    rng = np.random.default_rng(0)
    x = original_mixture(rng.normal(size=total), label="x")
    overlap = np.zeros(total)
    start = total // 4
    overlap[start:start + run_samples] = 1.0
    x_O = x.masked(overlap, label="x_O")

    activity = np.zeros((2, total))
    activity[0, :start + run_samples] = 1.0
    activity[1, start:] = 1.0
    solo = np.zeros((2, total))
    solo[0, :start] = 1.0
    solo[1, start + run_samples:] = 1.0
    return x, x_O, activity, solo


def _embeddings():
    return (np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            np.array([[0.001] * 3, [0.001] * 3]))


class TestShortOverlapDoesNotCrash:
    def test_a_two_sample_overlap_is_skipped_not_fatal(self):
        x, x_O, activity, solo = _scene_with_overlap_run(2)
        embeddings, variances = _embeddings()
        encoder = _ExplodingEncoder()

        final, rounds = refine_embeddings(
            x, x_O, activity, solo, embeddings, variances,
            _PassthroughExtractor(), encoder, SAMPLE_RATE, rounds=1, **GATE,
        )

        assert encoder.calls == 0, "the encoder was handed a clip it cannot embed"
        np.testing.assert_allclose(final, embeddings)

    def test_a_long_overlap_is_still_refined(self):
        """The floor must not become a blanket disable."""
        x, x_O, activity, solo = _scene_with_overlap_run(2000)  # 250 ms
        embeddings, variances = _embeddings()
        encoder = _ExplodingEncoder()

        _, rounds = refine_embeddings(
            x, x_O, activity, solo, embeddings, variances,
            _PassthroughExtractor(), encoder, SAMPLE_RATE, rounds=1, **GATE,
        )
        assert encoder.calls > 0
        assert any(r is not None and r.reason != "overlap_clip_too_short"
                   for r in rounds[0])


class TestSkipIsDistinguishableFromRejection:
    """`accepted=False` alone cannot tell the two apart — the reason must.

    Conflating them would hide how much real boundary jitter costs refinement,
    reporting it as ordinary gate rejection instead.
    """

    def test_skipped_speakers_carry_their_own_reason(self):
        x, x_O, activity, solo = _scene_with_overlap_run(2)
        embeddings, variances = _embeddings()

        _, rounds = refine_embeddings(
            x, x_O, activity, solo, embeddings, variances,
            _PassthroughExtractor(), _ExplodingEncoder(), SAMPLE_RATE,
            rounds=1, **GATE,
        )
        reasons = [r.reason for r in rounds[0] if r is not None]
        assert reasons, "a skip must be RECORDED, not left as None"
        assert all(r == "overlap_clip_too_short" for r in reasons), reasons

    def test_reason_is_not_the_generic_rejection_label(self):
        x, x_O, activity, solo = _scene_with_overlap_run(2)
        embeddings, variances = _embeddings()
        _, rounds = refine_embeddings(
            x, x_O, activity, solo, embeddings, variances,
            _PassthroughExtractor(), _ExplodingEncoder(), SAMPLE_RATE,
            rounds=1, **GATE,
        )
        for r in rounds[0]:
            if r is not None:
                assert r.reason not in ("accepted", "margin", "rejected")


class TestOptOutForToyScenes:
    def test_min_clip_ms_zero_restores_the_old_behaviour(self):
        """Phase 2's unit tests use a 5-sample scene and a fake encoder."""
        x, x_O, activity, solo = _scene_with_overlap_run(2)
        embeddings, variances = _embeddings()

        class _AnyLengthEncoder(_ExplodingEncoder):
            MIN_SAMPLES = 0

        encoder = _AnyLengthEncoder()
        refine_embeddings(
            x, x_O, activity, solo, embeddings, variances,
            _PassthroughExtractor(), encoder, SAMPLE_RATE,
            rounds=1, min_clip_ms=0.0, **GATE,
        )
        assert encoder.calls > 0
