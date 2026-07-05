"""Derive oracle speaker-active spans from the clean isolated sources.

WSJ0-2mix and LibriMix ship no RTTM, but Phase 0 uses *oracle* diarization —
and here the clean per-speaker sources ARE the ground truth. Where source ``i``
has energy, speaker ``i`` is talking; where it is silent (the zero-padding from
staggered placement, or natural pauses), they are not. A short-window energy
threshold recovers exactly the solo/overlap structure the mixer built in.

This yields :class:`~dagger.diarize.oracle.Segment` objects, so the rest of the
path (``activity_matrix`` -> ``solo_overlap_regions``) is reused unchanged.
"""

from __future__ import annotations

import numpy as np

from dagger.diarize.oracle import Segment


def _frame_energy(samples: np.ndarray, win: int) -> np.ndarray:
    """Per-sample smoothed energy: a box-average of ``samples**2`` over ``win``."""
    power = np.asarray(samples, dtype=np.float64) ** 2
    if win <= 1:
        return power
    box = np.ones(win, dtype=np.float64) / win
    return np.convolve(power, box, mode="same")


def active_mask(
    source: np.ndarray,
    sample_rate: int,
    *,
    win_ms: float = 25.0,
    threshold_db: float = -40.0,
    min_dur_ms: float = 50.0,
) -> np.ndarray:
    """Binary activity mask for one clean source, of shape ``[T]``.

    A frame is active when its smoothed energy is within ``threshold_db`` of the
    source's peak energy. Active runs shorter than ``min_dur_ms`` are dropped and
    silent gaps shorter than ``min_dur_ms`` are filled, so a few dropout samples
    inside a word don't shatter one utterance into many segments.
    """
    source = np.asarray(source, dtype=np.float64)
    n = source.shape[0]
    win = max(1, int(round(win_ms / 1000.0 * sample_rate)))
    energy = _frame_energy(source, win)

    peak = float(energy.max()) if n else 0.0
    if peak <= 0.0:
        return np.zeros(n, dtype=np.float64)  # a silent source is never active

    floor = peak * (10.0 ** (threshold_db / 10.0))  # energy is power -> /10, not /20
    mask = (energy >= floor).astype(np.float64)

    min_run = max(1, int(round(min_dur_ms / 1000.0 * sample_rate)))
    mask = _fill_short_runs(mask, min_run, value=0.0)  # bridge tiny gaps
    mask = _fill_short_runs(mask, min_run, value=1.0)  # drop tiny blips
    return mask


def _fill_short_runs(mask: np.ndarray, min_run: int, value: float) -> np.ndarray:
    """Flip runs of ``value`` shorter than ``min_run`` to the opposite value."""
    mask = mask.copy()
    n = mask.shape[0]
    i = 0
    while i < n:
        j = i
        while j < n and mask[j] == value:
            j += 1
        if j > i and (j - i) < min_run:
            mask[i:j] = 1.0 - value
        i = max(j, i + 1)
    return mask


def _mask_to_segments(mask: np.ndarray, speaker: str, sample_rate: int) -> list[Segment]:
    """Convert a binary mask into contiguous :class:`Segment` spans (seconds)."""
    segments: list[Segment] = []
    n = mask.shape[0]
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            start = i / sample_rate
            duration = (j - i) / sample_rate
            segments.append(Segment(speaker=speaker, start=start, duration=duration))
            i = j
        else:
            i += 1
    return segments


def segments_from_sources(
    sources: np.ndarray,
    speakers: list[str],
    sample_rate: int,
    **vad_kwargs,
) -> list[Segment]:
    """Oracle segments for every speaker, derived from the clean sources.

    ``sources`` is ``[S, T]`` with row ``i`` aligned to ``speakers[i]`` (the
    padded, aligned output of :func:`dagger.data.mixing.mix_sources`). Returns a
    flat list of :class:`Segment` across all speakers, ready for
    :func:`dagger.diarize.oracle.activity_matrix`.

    This is a VAD *estimate*, not exact ground truth: a source's quiet trailing
    consonants or soft onsets can fall below ``threshold_db`` while still being
    non-zero, leaking a faint residual across a solo/overlap boundary. When the
    exact placement window is known (:func:`dagger.data.mixing.mix_sources`'s
    ``offsets``/lengths, as in the LibriMix and WSJ0-2mix loaders) prefer
    :func:`segments_from_placement`, which has no such leakage. Reach for this
    function only when a corpus gives you clean per-speaker tracks without known
    placement (e.g. real solo/overlap reference recordings).
    """
    sources = np.asarray(sources, dtype=np.float64)
    if sources.shape[0] != len(speakers):
        raise ValueError("sources rows must match number of speakers.")
    segments: list[Segment] = []
    for i, speaker in enumerate(speakers):
        mask = active_mask(sources[i], sample_rate, **vad_kwargs)
        segments.extend(_mask_to_segments(mask, speaker, sample_rate))
    return segments


def segments_from_placement(
    offsets: list[int],
    lengths: list[int],
    speakers: list[str],
    sample_rate: int,
) -> list[Segment]:
    """Exact oracle segments from known placement windows -- no VAD, no leakage.

    For mixtures built by :func:`dagger.data.mixing.mix_sources`, speaker ``i``
    occupies exactly ``[offsets[i], offsets[i] + lengths[i])`` and is exactly
    zero outside it. Using that literal window (instead of estimating activity
    from signal energy, as :func:`segments_from_sources` does) means a solo
    region's copy is bit-exact: no quiet real content from a neighboring
    speaker's clip can be mistaken for silence and leak across the boundary.
    """
    if not (len(offsets) == len(lengths) == len(speakers)):
        raise ValueError("offsets, lengths, and speakers must have equal length.")
    segments: list[Segment] = []
    for speaker, offset, length in zip(speakers, offsets, lengths):
        if length <= 0:
            continue
        segments.append(
            Segment(
                speaker=speaker,
                start=offset / sample_rate,
                duration=length / sample_rate,
            )
        )
    return segments
