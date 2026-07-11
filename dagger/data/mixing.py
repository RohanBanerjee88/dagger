"""On-the-fly mixing shared by both corpora (CLAUDE.md §5, storage-lean).

Only the *source* utterances live on the mounted volume; the one-channel mixture
is summed here at load time and never written to disk. Both LibriMix (per-source
gains from a metadata CSV) and WSJ0-2mix (per-source gains from the listed SNRs)
reduce to the same operation: scale each source, place it at an offset, and sum.

Phase 0 mixes *staggered*, not fully overlapped: each successive utterance starts
partway through the previous one (:func:`stagger_offsets`). That is what exercises
the solo->copy / overlap->extract split — a fully-overlapped 2mix has ~no solo
region. Plain staggering only guarantees per-speaker solo time for 2 speakers;
for 3+ it depends on random length ratios, so the loaders pass ``min_solo`` to
make the guarantee unconditional (CLAUDE.md Phase 1 "KNOWN ISSUE").
"""

from __future__ import annotations

import numpy as np


def db_to_linear(gain_db: float) -> float:
    """Convert a gain in dB to a linear amplitude multiplier (``10**(dB/20))``."""
    return float(10.0 ** (float(gain_db) / 20.0))


def stagger_offsets(lengths: list[int], overlap: float, min_solo: int = 0) -> list[int]:
    """Start offsets (samples) that overlap each utterance with the previous one.

    ``overlap`` is the fraction of the *previous* utterance that the next one
    overlaps: ``0.0`` places utterances back-to-back (pure turn-taking, no
    overlap), ``1.0`` starts them all together (fully overlapped). Phase 0 uses a
    middling value so both solo and overlap regions exist.

    ``min_solo`` (samples, ``0`` = legacy behavior) guarantees every speaker a
    contiguous solo window of at least ``min(min_solo, own length)``. Plain chain
    staggering ties solo time to *random length ratios* — with ``overlap: 0.5``
    a 3-mix middle speaker is solo only when its utterance outlasts the first
    one, which starves ~70–80% of Libri3Mix scenes at enrollment (CLAUDE.md
    Phase 1 "KNOWN ISSUE"). With ``min_solo`` set, each start is pushed just
    late enough that (a) the previous speaker keeps a solo window before the
    next one begins and (b) each utterance outlasts the previous one by its own
    solo window. The guarantee takes precedence over ``overlap``: adjacent
    short utterances may end up with less overlap than requested, or none.

    Offsets are deterministic (no RNG) so a scene is reproducible from its config.
    """
    if not 0.0 <= overlap <= 1.0:
        raise ValueError(f"overlap must be in [0, 1], got {overlap}.")
    if min_solo < 0:
        raise ValueError(f"min_solo must be >= 0, got {min_solo}.")
    offsets = [0]
    prev_prev_end = 0  # end of the utterance two back (0 while there isn't one)
    for i in range(1, len(lengths)):
        prev_len = int(lengths[i - 1])
        own_len = int(lengths[i])
        prev_start = offsets[-1]
        prev_end = prev_start + prev_len
        # the requested stagger: (1 - overlap) into the previous utterance
        start = prev_start + int(round((1.0 - overlap) * prev_len))
        if min_solo > 0:
            # A speaker shorter than min_solo can at best be solo for its whole
            # length.
            prev_solo = min(min_solo, prev_len)
            own_solo = min(min_solo, own_len)
            # Earliest sample where the previous speaker can be solo: everyone
            # before it has ended (ends are non-decreasing under these
            # constraints).
            prev_solo_floor = max(prev_prev_end, prev_start)
            start = max(
                start,
                # leave the previous speaker its solo window before this one
                # begins
                prev_solo_floor + prev_solo,
                # end late enough that this speaker gets a solo tail of its own
                prev_end + own_solo - own_len,
            )
        offsets.append(start)
        prev_prev_end = prev_end
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
