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


#: Widest gain correction :func:`match_level_to_mixture` will apply, as a ratio.
#: A projection this far from 1 means the estimate barely correlates with the
#: mixture, where the least-squares scalar is noise rather than a calibration.
MAX_RESCALE = 8.0


def match_level_to_mixture(
    g_out: np.ndarray, x_O: np.ndarray, w_Oi: np.ndarray
) -> np.ndarray:
    """Rescale ``G``'s output to the level implied by the overlap mixture.

    Measured 2026-08-23: ``G`` emits the overlap region at a median **2.86x**
    the true source amplitude (``level_error_db`` +9.14). The cause is not a bug
    in the audio path -- it is that ``si_sdr_loss`` is scale-invariant
    (``dagger/losses/sisdr.py``), so **nothing in training ever constrained the
    output level**, and it drifted. Every SI-SDR in this project is likewise
    scale-invariant, so the error was structurally invisible until the
    whole-track and per-depth numbers were compared.

    The correction uses only the mixture, never ground truth, so it is
    deployable. Over the emitted region it takes the least-squares scalar::

        c = <g, x_O> / <g, g>

    which is exact in the ideal case: ``x_O = s_i + sum_{j!=i} s_j`` and the
    other speakers are approximately orthogonal to ``s_i``, so if
    ``g ~ k*s_i + n`` then ``<g, x_O> ~ k*||s_i||^2`` and ``c ~ 1/k``. With real
    noise it shrinks slightly toward zero (it is the MMSE gain, not the exact
    inverse), which is the conservative direction: under-correcting leaves a
    smaller error than over-correcting.

    Applied only where ``w_Oi > 0``. The solo path is a verbatim copy of the
    mixture and is already at unity gain -- rescaling it could only introduce an
    error. Refuses to act (returns ``g_out`` unchanged) when the projection is
    degenerate: a non-positive numerator would flip the waveform's sign, and a
    ratio beyond :data:`MAX_RESCALE` means the estimate is too weakly correlated
    with the mixture for the scalar to mean anything.
    """
    g = np.asarray(g_out, dtype=np.float64)
    mixture = np.asarray(x_O, dtype=np.float64)
    mask = np.asarray(w_Oi, dtype=np.float64) > 0
    if not mask.any():
        return g

    num = float(np.dot(g[mask], mixture[mask]))
    den = float(np.dot(g[mask], g[mask]))
    if den <= 0.0 or num <= 0.0:
        return g
    c = num / den
    if not (1.0 / MAX_RESCALE) <= c <= MAX_RESCALE:
        return g
    return g * c


def reconstruct_speaker(
    x: TrackedSignal,
    x_O: TrackedSignal,
    activity_i: np.ndarray,
    solo_i: np.ndarray,
    embedding: np.ndarray,
    extractor: Extractor,
    fade: int = 0,
    rescale_to_mixture: bool = False,
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
    g_out = np.asarray(extractor.extract(x_O, embedding), dtype=np.float64)  # guarded
    if rescale_to_mixture:
        g_out = match_level_to_mixture(g_out, np.asarray(x_O, dtype=np.float64), w_Oi)
    return x_samples * w_Ei + g_out * w_Oi


def reconstruct_all(
    x: TrackedSignal,
    x_O: TrackedSignal,
    activity: np.ndarray,
    solo: np.ndarray,
    embeddings: np.ndarray | None,
    extractor: Extractor,
    fade: int = 0,
    rescale_to_mixture: bool = False,
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
            rescale_to_mixture=rescale_to_mixture,
        )
    return outputs
