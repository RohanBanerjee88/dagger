"""Soft-mask stitching (partition of unity) for the audio path.

For each speaker ``i`` we combine two contributions over its active region:

* solo frames ``E_i`` — copied straight from the mixture ``x`` (running a network
  on already-clean audio only adds artifacts; CLAUDE.md §2 "copy, don't
  separate");
* overlap frames — produced by the extractor ``G(x_O, e_i)`` from the *untouched*
  overlap mixture ``x_O``.

The two windows crossfade smoothly at the solo↔overlap seam and satisfy the
partition of unity ``w_Ei + w_Oi = a_i`` everywhere, so there is no click and
the network always sees some context (CLAUDE.md §2 "soft masks at seams").
"""

from __future__ import annotations

import numpy as np

from dagger.audio.provenance import TrackedSignal
from dagger.extract.base import Extractor


def crossfade_windows(
    solo_i: np.ndarray,
    activity_i: np.ndarray,
    fade: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the solo window ``w_Ei`` and overlap window ``w_Oi`` for speaker ``i``.

    ``solo_i`` marks frames where ``i`` is active *alone*; ``activity_i`` marks all
    frames where ``i`` is active. The returned windows satisfy, at every sample:

        ``w_Ei + w_Oi == activity_i``   (partition of unity over the active region)

    with a linear crossfade of half-width ``fade`` samples at each solo↔overlap
    boundary. ``fade == 0`` gives hard masks (``w_Ei == solo_i``).
    """
    solo_i = np.asarray(solo_i, dtype=np.float64)
    activity_i = np.asarray(activity_i, dtype=np.float64)

    if fade <= 0:
        w_Ei = solo_i.copy()
    else:
        # Box-average the solo preference over the *active* support only, so the
        # ramp appears at solo↔overlap seams without bleeding across activity
        # edges. numerator/denominator are both smoothed over the same window.
        box = np.ones(2 * fade + 1, dtype=np.float64)
        numer = np.convolve(solo_i, box, mode="same")
        denom = np.convolve(activity_i, box, mode="same")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(denom > 0, numer / denom, 0.0)
        w_Ei = np.clip(ratio, 0.0, 1.0) * activity_i

    w_Oi = activity_i - w_Ei
    return w_Ei, w_Oi


def reconstruct_speaker(
    x: TrackedSignal,
    x_O: TrackedSignal,
    activity_i: np.ndarray,
    solo_i: np.ndarray,
    embedding: np.ndarray,
    extractor: Extractor,
    fade: int = 0,
) -> np.ndarray:
    """Reconstruct one speaker's waveform per CLAUDE.md §1.

    ``s_hat_i = x·w_Ei + G(x_O, e_i)·w_Oi``. The extractor is fed ``x_O`` (an
    original-mixture :class:`TrackedSignal`); it will raise if handed a residual.

    Known Phase 1 limitation (CLAUDE.md §5 Phase 1 red flags: "G receiving
    x·1_Oi with hard masks (fix later, but note it)"): ``x_O`` is built once by
    the caller via a hard binary ``overlap`` mask and shared, unchanged, across
    every speaker here -- ``G`` never sees context beyond the overlap region.
    This is scheduled to be revisited in Phase 2 (the reconstruction-quality
    phase), not fixed here.
    """
    w_Ei, w_Oi = crossfade_windows(solo_i, activity_i, fade=fade)
    x_samples = np.asarray(x, dtype=np.float64)
    g_out = extractor.extract(x_O, embedding)  # guarded against residual inputs
    return x_samples * w_Ei + np.asarray(g_out, dtype=np.float64) * w_Oi


def reconstruct_all(
    x: TrackedSignal,
    x_O: TrackedSignal,
    activity: np.ndarray,
    solo: np.ndarray,
    embeddings: np.ndarray | None,
    extractor: Extractor,
    fade: int = 0,
) -> np.ndarray:
    """Reconstruct every speaker; returns an array of shape ``[S, T]``.

    ``embeddings`` may be ``None`` in Phase 0 (the null extractor ignores them).
    Every speaker is extracted from the same untouched ``x_O`` — the accumulation
    -free property (CLAUDE.md §1): no speaker's output depends on another's.
    """
    num_speakers = activity.shape[0]
    outputs = np.zeros_like(activity, dtype=np.float64)
    for i in range(num_speakers):
        emb = None if embeddings is None else embeddings[i]
        outputs[i] = reconstruct_speaker(
            x=x,
            x_O=x_O,
            activity_i=activity[i],
            solo_i=solo[i],
            embedding=emb,
            extractor=extractor,
            fade=fade,
        )
    return outputs
