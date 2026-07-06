"""Scale-invariant signal-to-distortion ratio (SI-SDR).

SI-SDR projects the estimate onto the target to remove an arbitrary gain, then
reports the target-to-error power ratio in dB. Higher is better; a perfect
estimate gives ``+inf``.
"""

from __future__ import annotations

import itertools

import numpy as np

_EPS = 1e-8


def si_sdr(estimate: np.ndarray, target: np.ndarray) -> float:
    """SI-SDR in dB between ``estimate`` and ``target`` (1-D waveforms).

    Returns ``+inf`` for a perfect estimate, ``-inf`` for a silent estimate
    against real target energy (e.g. an extractor that output nothing), and
    ``nan`` if the target is silent (SI-SDR is undefined against a zero
    reference).
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    target_energy = float(np.dot(target, target))
    if target_energy < _EPS:
        return float("nan")

    # A silent estimate makes scale/projection/noise all exactly zero below --
    # 10*log10(0/0) is indeterminate, not "perfect". Total silence against a
    # real target is a total failure, so score it -inf before that ambiguity
    # can masquerade as +inf.
    estimate_energy = float(np.dot(estimate, estimate))
    if estimate_energy < _EPS:
        return float("-inf")

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


def si_sdr_best_permutation(
    estimates: np.ndarray,
    targets: np.ndarray,
) -> tuple[list[float], tuple[int, ...]]:
    """Best-permutation per-speaker SI-SDR for order-unconstrained output.

    Blind separation (:class:`~dagger.extract.blind.BlindSeparator`) has no
    fixed output order, unlike the proposed extractor (whose output order
    always matches the embedding it was conditioned on). ``estimates``/
    ``targets`` are ``[S, T]``. Returns ``(per_speaker_si_sdr, perm)`` for the
    permutation of ``estimates`` rows that maximizes total SI-SDR (``nan``
    values, from a silent target, are treated as 0 for the purpose of ranking
    permutations but reported as-is).
    """
    estimates = np.asarray(estimates)
    targets = np.asarray(targets)
    num_speakers = targets.shape[0]

    best_perm = tuple(range(num_speakers))
    best_scores: list[float] | None = None
    best_total = float("-inf")
    for perm in itertools.permutations(range(num_speakers)):
        scores = [si_sdr(estimates[perm[i]], targets[i]) for i in range(num_speakers)]
        total = sum(0.0 if np.isnan(s) else s for s in scores)
        if total > best_total:
            best_total = total
            best_scores = scores
            best_perm = perm
    return best_scores or [], best_perm
