"""Scale-invariant signal-to-distortion ratio (SI-SDR).

SI-SDR projects the estimate onto the target to remove an arbitrary gain, then
reports the target-to-error power ratio in dB. Higher is better; a perfect
estimate gives ``+inf``.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    """SI-SDR in dB between ``estimate`` and ``target`` (1-D waveforms).

    Returns ``+inf`` for a perfect estimate and ``nan`` if the target is silent
    (SI-SDR is undefined against a zero reference).
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    target_energy = float(np.dot(target, target))
    if target_energy < _EPS:
        return float("nan")

    scale = float(np.dot(estimate, target)) / (target_energy + _EPS)
    projection = scale * target
    noise = estimate - projection

    noise_energy = float(np.dot(noise, noise))
    if noise_energy < _EPS:
        return float("inf")
    return 10.0 * np.log10(float(np.dot(projection, projection)) / noise_energy)


def si_sdr_regionwise(
    estimate: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> float:
    """SI-SDR computed only over samples where ``mask`` is truthy.

    Used to score solo vs. overlap regions separately — the plumbing check in
    Phase 0 is that solo regions score essentially perfectly.
    """
    mask = np.asarray(mask).astype(bool)
    if not mask.any():
        return float("nan")
    return si_sdr(np.asarray(estimate)[mask], np.asarray(target)[mask])
