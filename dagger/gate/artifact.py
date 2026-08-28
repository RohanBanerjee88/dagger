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


def spectral_flatness(
        estimate: np.ndarray,
        n_fft: int = 512,
        hop: int = 128,
        *,
        min_energy_db: float | None = None
) -> float:
    """Mean spectral flatness of ``estimate`` over time.

    Spectral flatness is a measure of how noise-like a signal is. A value of 1
    indicates a flat spectrum (white noise), while a value closer to 0 indicates
    a more tonal signal (like speech). High spectral flatness in the estimate
    suggests artifacts rather than clean speech.

    Args:
        estimate: The audio signal to analyze.
        n_fft: The number of FFT points for the STFT.
        hop: The hop length for the STFT.
        min_energy_db: If provided, keep only frames whose energy is within this
            many dB of the clip's PEAK frame energy (so it is relative, not an
            absolute level) -- the same peak-relative convention
            ``dagger.data.activity.active_mask`` uses, so both gate checks agree
            on what "a frame with speech in it" means. ``None`` reproduces the
            pre-2026-08-26 whole-track mean exactly. A ``max_artifact_score``
            tuned WITH gating is a threshold on a different quantity than one
            tuned without it, so the two must travel together --
            ``scripts/tune_gate.py`` refuses to recommend a threshold unless the
            config states ``artifact_min_energy_db`` explicitly.


    Returns:
        The mean spectral flatness of the estimate.
    """
    x = np.asarray(estimate, dtype=np.float64)
    n = x.shape[0]
    if n < n_fft:
        return float("nan")
    eps = 1e-10
    window = np.hanning(n_fft)
    starts = list(range(0, n - n_fft + 1, hop))

    if min_energy_db is not None:
        # Energy in the RAW frame, not the windowed one: active mask
        # measures box-averaged power on the raw samples, and a Hann taper would make the 
        # same speech frame look quieter purely by position within the hop
        energies = np.array([float(np.mean(x[s : s + n_fft] ** 2)) for s in starts])
        floor_ref = float(energies.max())
        if floor_ref <= 0.0:
            # Every frame is digital silence, so the mean spectral flatness 
            # which is an "artifact-like" measurement. 
            return 1.0
        floor = floor_ref * (10.0 ** (min_energy_db / 10.0)) # energy is power -> /10
        starts = [s for s, e in zip(starts, energies) if e >= floor]
    flatness_per_frame = []
    for start in starts:
        frame = x[start : start + n_fft] * window
        spec = np.abs(np.fft.rfft(frame)) + eps
        geometric_mean = float(np.exp(np.mean(np.log(spec))))
        arithmetic_mean = float(np.mean(spec))
        flatness_per_frame.append(geometric_mean / arithmetic_mean)
    if not flatness_per_frame:
        return float("nan")
    return float(np.mean(flatness_per_frame))