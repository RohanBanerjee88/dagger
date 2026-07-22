"""VAD coverage + artifact score components of the confidence gate (CLAUDE.md §5 Phase 2).

Two independent, cheap, dependency-free checks on an extracted estimate ``ŝ_i``:

* ``vad_coverage`` -- does the estimate actually contain detected speech where
  the diarizer says speaker ``i`` should be active? A near-zero coverage means
  the extractor produced near-silence instead of the target speaker.
* ``spectral_flatness`` -- a coarse proxy for mask-based-separation artifacts
  (musical noise): high spectral flatness (closer to white noise) in the
  estimate suggests artifacts rather than clean speech.
"""

from __future__ import annotations

import numpy as np

from dagger.data.activity import active_mask


def vad_coverage(
    estimate: np.ndarray,
    expected_active: np.ndarray,
    sample_rate: int,
    **vad_kwargs,
) -> float:
    """Fraction of ``expected_active`` samples where ``estimate`` is detected as active.

    ``expected_active`` is the speaker's oracle activity mask over the region
    being scored (e.g. the overlap portion of ``activity_i``). ``nan`` if
    ``expected_active`` has no active samples (nothing to check coverage over).
    """
    expected = np.asarray(expected_active).astype(bool)
    if not expected.any():
        return float("nan")
    estimate = np.asarray(estimate, dtype=np.float64)
    if estimate.shape[0] == 0:
        return float("nan")
    # active_mask's default win_ms=25 assumes a clip of at least that many
    # samples; a short overlap-region clip (e.g. a Phase 2 refinement round's
    # run) shorter than the analysis window would otherwise make
    # np.convolve(..., mode="same") return a window-sized array instead of one
    # matching `estimate`. Clamp the window to the clip's own duration.
    clip_ms = 1000.0 * estimate.shape[0] / sample_rate
    vad_kwargs = dict(vad_kwargs)
    vad_kwargs["win_ms"] = min(vad_kwargs.get("win_ms", 25.0), clip_ms)
    detected = active_mask(estimate, sample_rate, **vad_kwargs).astype(bool)
    return float(np.mean(detected[expected]))


def spectral_flatness(estimate: np.ndarray, n_fft: int = 512, hop: int = 128) -> float:
    """Mean spectral flatness (geometric mean / arithmetic mean of the magnitude
    spectrum, a.k.a. Wiener entropy) across frames, in ``[0, 1]``.

    Closer to 1 means noise-like (flat spectrum -- a proxy for mask-based
    separation artifacts / musical noise); closer to 0 means tonal/speech-like.
    ``nan`` if the estimate is shorter than one frame.
    """
    x = np.asarray(estimate, dtype=np.float64)
    n = x.shape[0]
    if n < n_fft:
        return float("nan")
    eps = 1e-10
    window = np.hanning(n_fft)
    flatness_per_frame = []
    for start in range(0, n - n_fft + 1, hop):
        frame = x[start : start + n_fft] * window
        spec = np.abs(np.fft.rfft(frame)) + eps
        geo_mean = float(np.exp(np.mean(np.log(spec))))
        arith_mean = float(np.mean(spec))
        flatness_per_frame.append(geo_mean / arith_mean)
    if not flatness_per_frame:
        return float("nan")
    return float(np.mean(flatness_per_frame))
