"""On-the-fly mixing shared by both corpora (CLAUDE.md §5, storage-lean).

Only the *source* utterances live on the mounted volume; the one-channel mixture
is summed here at load time and never written to disk. Both LibriMix (per-source
gains from a metadata CSV) and WSJ0-2mix (per-source gains from the listed SNRs)
reduce to the same operation: scale each source, place it at an offset, and sum.

Phase 0 mixes *staggered*, not fully overlapped: each successive utterance starts
partway through the previous one (:func:`stagger_offsets`), so every mixture has a
solo lead-in, an overlap middle, and a solo tail. That is what exercises the
solo->copy / overlap->extract split — a fully-overlapped 2mix has ~no solo region.
"""

from __future__ import annotations

import numpy as np


def db_to_linear(gain_db: float) -> float:
    """Convert a gain in dB to a linear amplitude multiplier (``10**(dB/20))``."""
    return float(10.0 ** (float(gain_db) / 20.0))


def stagger_offsets(lengths: list[int], overlap: float) -> list[int]:
    """Start offsets (samples) that overlap each utterance with the previous one.

    ``overlap`` is the fraction of the *previous* utterance that the next one
    overlaps: ``0.0`` places utterances back-to-back (pure turn-taking, no
    overlap), ``1.0`` starts them all together (fully overlapped). Phase 0 uses a
    middling value so both solo and overlap regions exist.

    Offsets are deterministic (no RNG) so a scene is reproducible from its config.
    """
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be in [0, 1], got {overlap}.")
    offsets = [0]
    for prev_len in lengths[:-1]:
        # next utterance starts after (1 - overlap) of the previous one
        step = int(round((1.0 - overlap) * prev_len))
        offsets.append(offsets[-1] + step)
    return offsets


def mix_sources(
    sources: list[np.ndarray],
    *,
    gains: list[float] | None = None,
    offsets: list[int] | None = None,
    length_mode: str = "max",
    length: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale, place, and sum variable-length sources into one mixture.

    * ``gains``   — linear amplitude per source (default all 1.0).
    * ``offsets`` — start sample of each source within the mixture (default 0).
    * ``length``  — explicit output length ``T``; otherwise derived from
      ``length_mode``: ``"max"`` pads every source to the longest placed extent
      (keeps the solo tail — the Phase 0 default), ``"min"`` truncates to the
      shortest placed extent.

    Returns ``(padded_sources, mixture)`` with shapes ``[S, T]`` and ``[T]``. The
    returned per-source rows are the aligned, gain-scaled, length-normalized
    sources — i.e. the ground-truth targets that sum exactly to ``mixture``.
    """
    num = len(sources)
    if num == 0:
        raise ValueError("mix_sources needs at least one source.")
    gains = [1.0] * num if gains is None else list(gains)
    offsets = [0] * num if offsets is None else list(offsets)
    if not (len(gains) == len(offsets) == num):
        raise ValueError("gains and offsets must match the number of sources.")

    ends = [int(offsets[i]) + len(sources[i]) for i in range(num)]
    if length is not None:
        out_len = int(length)
    elif length_mode == "max":
        out_len = max(ends)
    elif length_mode == "min":
        out_len = min(ends)
    else:
        raise ValueError(f"length_mode must be 'min' or 'max', got {length_mode!r}.")

    padded = np.zeros((num, out_len), dtype=np.float64)
    for i in range(num):
        src = np.asarray(sources[i], dtype=np.float64) * float(gains[i])
        start = int(offsets[i])
        # place [start : start+len(src)] clipped to [0, out_len)
        lo = max(0, start)
        hi = min(out_len, start + len(src))
        if hi <= lo:
            continue
        padded[i, lo:hi] = src[lo - start : hi - start]

    mixture = padded.sum(axis=0)
    return padded, mixture
