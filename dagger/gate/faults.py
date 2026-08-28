"""Deliberate corruptions of an extractor's overlap-region output.

**NOT DEPLOYABLE. Tuning-time only.** Nothing here may ever run in a reported
system. These functions manufacture the labelled fault populations that
``scripts/tune_gate.py`` sweeps ``min_vad_coverage`` and ``max_artifact_score``
against; the entire deliverable is two floats in a yaml. Same status as
:mod:`dagger.refine.oracle_ceiling`, and deliberately NOT re-exported from
``dagger.gate.__init__`` so that an eval-path import stands out.

Why they are needed at all
--------------------------
A threshold cannot be placed without something to detect. ``max_mean_variance``
only moved 0.05 -> 1e-4 once the contaminated-enrollment fixture gave ``V_i`` a
population to detect against. ``min_vad_coverage`` and ``max_artifact_score``
have never had one -- which is exactly why they are the two checks still untuned
after 10,950 recorded gate decisions, and why "0 firings in 10,950" has never
been evidence either way (CLAUDE.md §9: "'It never fires' is not evidence that a
check is useless" -- that inference was wrong by 500x once already).

Contract
--------
Every function here has the same shape::

    corrupt(signal[T], region[T] bool, severity, ...) -> [T]

``signal`` is ``G``'s output ``g_out``, FULL-LENGTH -- not just the overlap
samples. ``region`` is ``w_Oi > 0``, the support ``G`` is actually responsible
for; samples outside it are returned untouched. That matters twice: the stitch
multiplies them by ~0 anyway, and a severity like "50% dropped" must mean half
of the REGION, not half of a track that is mostly other speakers' turns.

Corrupting ``g_out`` rather than the finished estimate is deliberate. The
estimate is ``x*w_Ei + g_out*w_Oi``; its solo half is a straight copy of the
mixture -- a Phase 0 guarantee ``G`` cannot break. Corrupting the whole track
would model "the pipeline broke", which is not the failure the gate exists to
catch.

Every fixture is GRADED. A binary "emit silence" fault scores Youden's J = 1.0
at every candidate threshold and therefore places none -- the same degenerate
sweep oracle-region ``V_i`` produced, where a healthy population pinned at
exactly 0 separated perfectly from anything at all.
"""

from __future__ import annotations

import numpy as np


def _rng(rng: np.random.Generator | None) -> np.random.Generator:
    return np.random.default_rng(0) if rng is None else rng


def drop_span(
    signal: np.ndarray,
    region: np.ndarray,
    fraction: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Zero a CONTIGUOUS run covering ``fraction`` of ``region``'s samples.

    Models an extractor that emits silence over part of the region it was asked
    for -- target dropout, the failure ``min_vad_coverage`` is written against.

    Contiguous *in the region's own index space*, not scattered:
    ``active_mask`` fills silent gaps shorter than ``min_dur_ms=50``, so
    sample-scattered dropout would be silently repaired by the very detector
    under test and the fixture would measure nothing. Contiguity in region-index
    space also means a fragmented region loses one stretch of the speaker's
    overlap speech rather than a stretch of wall-clock time that is mostly other
    speakers' turns.
    """
    out = np.asarray(signal, dtype=np.float64).copy()
    idx = np.flatnonzero(np.asarray(region, dtype=bool))
    if idx.size == 0 or fraction <= 0.0:
        return out
    count = int(round(min(float(fraction), 1.0) * idx.size))
    if count <= 0:
        return out
    start = 0 if count >= idx.size else int(_rng(rng).integers(0, idx.size - count + 1))
    out[idx[start : start + count]] = 0.0
    return out


def attenuate(signal: np.ndarray, region: np.ndarray, gain_db: float) -> np.ndarray:
    """Scale ``region`` by ``gain_db`` (negative attenuates).

    Models an extractor that emits the overlap far too quietly -- the mirror of
    the defect Q5 actually found, where ``G`` emits it at 2.86x too LOUD.

    **Detectable only because the peak reference includes the solo copy.**
    ``active_mask`` thresholds each frame against the CLIP'S OWN peak energy, so
    a uniform gain on the whole estimate is invisible to ``vad_coverage`` --
    scale-invariant, the same property that hid Q5's level error from every
    SI-SDR in this project for three phases (CLAUDE.md §7). A region-selective
    gain is a different thing: the loud, untouched solo half keeps the peak where
    it was, so a region pushed 30 dB down falls under the -40 dB floor. Do NOT
    "simplify" this into a whole-signal gain; that fixture is inert, and it would
    be inert in the quiet way -- a plausible number and nothing failing.
    """
    out = np.asarray(signal, dtype=np.float64).copy()
    mask = np.asarray(region, dtype=bool)
    out[mask] *= 10.0 ** (float(gain_db) / 20.0)  # amplitude -> /20, not /10
    return out


def add_noise(
    signal: np.ndarray,
    region: np.ndarray,
    snr_db: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add broadband Gaussian noise over ``region`` at ``snr_db`` relative to the
    signal's own energy there.

    Models the broadband residue a mask-based separator leaves behind. Raises
    spectral flatness monotonically -- measured 2026-08-26 on a speech-like
    fixture: clean 0.483 -> 20 dB 0.591 -> 10 dB 0.701 -> 0 dB 0.801, against
    pure white noise at 0.847.
    """
    out = np.asarray(signal, dtype=np.float64).copy()
    mask = np.asarray(region, dtype=bool)
    if not mask.any():
        return out
    power = float(np.mean(out[mask] ** 2))
    if power <= 0.0:
        return out
    noise = _rng(rng).normal(size=int(mask.sum()))
    noise *= np.sqrt(power / float(np.mean(noise**2)) / (10.0 ** (float(snr_db) / 10.0)))
    out[mask] += noise
    return out


def punch_holes(
    signal: np.ndarray,
    region: np.ndarray,
    punch_fraction: float,
    *,
    n_fft: int = 512,
    hop: int = 128,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Zero a random ``punch_fraction`` of STFT bins, resynthesize, and write the
    result back inside ``region`` -- MUSICAL NOISE, what mask-based separation
    actually produces.

    **The direction question this exists to answer.** Additive noise makes a
    spectrum flatter. Musical noise is SPARSE and tonal, which may make it
    PEAKIER. If the two families move ``spectral_flatness`` in opposite
    directions then no single ``max_artifact_score`` catches both, and the check
    needs REPLACING rather than tuning. No committed number can currently
    distinguish those cases, which is the whole reason this fixture is worth its
    extra complexity.

    Weighted overlap-add with a Hann analysis *and* synthesis window, divided by
    the accumulated ``w**2`` -- so resynthesis is exact when nothing is punched,
    regardless of whether the hop happens to satisfy COLA.
    ``tests/phase3/test_gate_faults.py`` asserts that identity, which is what
    makes a nonzero effect attributable to the punching rather than to the
    transform.
    """
    x = np.asarray(signal, dtype=np.float64)
    out = x.copy()
    mask = np.asarray(region, dtype=bool)
    n = x.shape[0]
    if n < n_fft or not mask.any() or punch_fraction <= 0.0:
        return out
    window = np.hanning(n_fft)
    generator = _rng(rng)
    acc = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    for start in range(0, n - n_fft + 1, hop):
        spec = np.fft.rfft(x[start : start + n_fft] * window)
        spec = spec * (generator.random(spec.shape[0]) >= punch_fraction)
        acc[start : start + n_fft] += np.fft.irfft(spec, n=n_fft) * window
        wsum[start : start + n_fft] += window**2
    # Only inside the region, and only where the overlap-add actually covered
    # the sample -- the first and last few samples see too little window weight
    # for the division to be meaningful.
    good = mask & (wsum > 1e-8)
    out[good] = acc[good] / wsum[good]
    return out
