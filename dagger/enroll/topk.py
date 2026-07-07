"""Top-K solo clips -> mean embedding e_bar_i (+ variance V_i).

Enrollment only ever draws from a speaker's *solo* region ``E_i`` (frames where
they are the only active speaker) -- CLAUDE.md §2's settled fact "solo regions
are clean" is what makes this a valid embedding source. Drawing from
``overlap`` by accident is one of Phase 1's named red flags; the assertions
below make that mistake fail loudly instead of silently contaminating ``e_bar_i``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dagger.enroll.encoder import SpeakerEncoder


class NoSoloRegionError(ValueError):
    """A speaker has no usable solo audio to enroll from.

    This is the one benign, expected-to-happen-sometimes enrollment failure
    (a speaker who's never alone long enough, plausible on multi-speaker
    corpora with high overlap fractions) -- callers may legitimately catch
    *this specific type* to skip a scene/speaker and continue. It is
    deliberately a different type from the plain :class:`ValueError` raised
    below for the overlap-contamination guard, which must never be silently
    caught: that one indicates a real diarization/oracle-region bug and
    should fail loudly (see this module's docstring).
    """


@dataclass(frozen=True)
class EnrollmentResult:
    """One speaker's enrollment: mean embedding, per-dimension variance, clip count.

    ``variance`` (``V_i``) is computed now even though it is unused until
    Phase 3's enrollment-rejection gate -- cheap to compute alongside the mean,
    and avoids retrofitting this dataclass later.
    """

    embedding: np.ndarray  # e_bar_i, [D]
    variance: np.ndarray  # V_i, [D]
    clip_count: int


def _solo_runs(solo_i: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``(start, end)`` sample runs where ``solo_i`` is truthy."""
    mask = np.asarray(solo_i).astype(bool)
    n = mask.shape[0]
    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def select_topk_solo_clips(
    mixture: np.ndarray,
    solo_i: np.ndarray,
    sample_rate: int,
    k: int = 3,
    min_clip_ms: float = 500.0,
    activity_i: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Return up to ``k`` longest solo clips for one speaker, longest first.

    ``solo_i`` must be the speaker's *solo* mask (never ``overlap``); if
    ``activity_i`` (that speaker's full activity mask) is also given, this
    asserts ``solo_i`` is a subset of it -- a direct guard against the
    "enrollment taken from overlap by accident" red flag. Runs shorter than
    ``min_clip_ms`` are dropped (too little audio for a stable embedding).
    """
    solo_i = np.asarray(solo_i)
    if not solo_i.any():
        raise NoSoloRegionError(
            "select_topk_solo_clips: solo_i has no active samples -- this "
            "speaker has no solo region to enroll from."
        )
    if activity_i is not None:
        activity_i = np.asarray(activity_i)
        if not np.all(solo_i.astype(bool) <= activity_i.astype(bool)):
            # Deliberately a plain ValueError, NOT NoSoloRegionError: this signals
            # a real diarization/oracle-region bug, not benign missing solo audio,
            # and must not be caught by skip-and-continue callers.
            raise ValueError(
                "select_topk_solo_clips: solo_i is not a subset of activity_i -- "
                "this looks like an overlap frame was mistaken for solo audio."
            )

    min_run = max(1, int(round(min_clip_ms / 1000.0 * sample_rate)))
    runs = [(start, end) for start, end in _solo_runs(solo_i) if (end - start) >= min_run]
    runs.sort(key=lambda run: run[1] - run[0], reverse=True)

    mixture = np.asarray(mixture)
    return [mixture[start:end] for start, end in runs[:k]]


def mean_embedding(
    clips: list[np.ndarray],
    sample_rate: int,
    encoder: SpeakerEncoder,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed each clip and return ``(mean, variance)`` across clips, shape ``[D]`` each."""
    if not clips:
        raise NoSoloRegionError("mean_embedding: no clips to embed.")
    embeddings = np.stack([encoder.embed(clip, sample_rate) for clip in clips], axis=0)
    return embeddings.mean(axis=0), embeddings.var(axis=0)


def enroll_speaker(
    mixture: np.ndarray,
    solo_i: np.ndarray,
    activity_i: np.ndarray,
    sample_rate: int,
    encoder: SpeakerEncoder,
    k: int = 3,
    min_clip_ms: float = 500.0,
) -> EnrollmentResult:
    """Build one speaker's :class:`EnrollmentResult` from their solo region."""
    clips = select_topk_solo_clips(
        mixture, solo_i, sample_rate, k=k, min_clip_ms=min_clip_ms, activity_i=activity_i
    )
    embedding, variance = mean_embedding(clips, sample_rate, encoder)
    return EnrollmentResult(embedding=embedding, variance=variance, clip_count=len(clips))
