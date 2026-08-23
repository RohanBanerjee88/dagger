"""Scale-invariant signal-to-distortion ratio (SI-SDR).

SI-SDR projects the estimate onto the target to remove an arbitrary gain, then
reports the target-to-error power ratio in dB. Higher is better; a perfect
estimate gives ``+inf``.
"""

from __future__ import annotations

import itertools
import math

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


def _region_energies(
    estimate: np.ndarray, target: np.ndarray
) -> tuple[float, float, float]:
    """``(scale, projection_energy, noise_energy)`` for one region.

    The three quantities :func:`si_sdr` computes before taking the ratio, kept
    separately so several regions can be pooled by *energy* (which is additive)
    rather than by dB (which is not).
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    target_energy = float(np.dot(target, target))
    if target_energy < _EPS:
        return float("nan"), 0.0, 0.0

    scale = float(np.dot(estimate, target)) / (target_energy + _EPS)
    projection = scale * target
    noise = estimate - projection
    return scale, float(np.dot(projection, projection)), float(np.dot(noise, noise))


def si_sdr_pooled_by_depth(
    estimate: np.ndarray,
    target: np.ndarray,
    depth: np.ndarray,
    min_depth: int = 1,
) -> float:
    """One number for the whole output that the per-depth tables can license.

    Fits the SI-SDR scale **per depth** — exactly as :func:`si_sdr_by_depth`
    already does — then pools the resulting error *energies* across depths,
    weighting each region by how much true speech it holds::

        10 * log10( sum_k T_k / sum_k (T_k / r_k) )

    where ``T_k`` is the target's energy at depth ``k`` and ``r_k`` is that
    depth's SI-SDR as a power ratio.

    Why not simply ``si_sdr(estimate, target)`` over the whole track: SI-SDR
    fits ONE scalar over whatever samples it is handed, and which samples you
    include decides what that scalar becomes (CLAUDE.md §7). When most of the
    track is a bit-exact solo copy, that scalar is pinned near 1 and a pure
    *level* error in the overlap region is charged at full price — while every
    per-depth row, fitting its own scalar, discounts the same error entirely.
    The two then disagree without either being wrong, and the whole-track number
    can land below every depth it supposedly summarises (measured on real data
    2026-08-23: -13.17 dB against constituents of +46.91 and +1.08).

    This function cannot do that. ``sum P / sum N`` is a weighted mediant of the
    per-depth ratios, so it is bounded by the smallest and largest of them, and
    it is invariant to a per-region gain. That is what makes it usable as an
    exchange rate — "does this configuration trade depth 1 for depth 2 at a
    profit?" — which is the question the un-stratified number was added to
    answer and cannot.

    It does NOT replace stratification (CLAUDE.md §6.4): report it beside the
    per-depth tables, never instead of them. Depth 0 (nobody active) is excluded
    for the same reason :func:`si_sdr_by_depth` excludes it — nothing was
    claimed there, so there is nothing to score.

    Returns ``nan`` when no depth ``>= min_depth`` carries target energy, and
    ``+inf`` when every scoreable region is perfect.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    depth = np.asarray(depth)

    # Weights come from the TARGET's energy per region, never the estimate's.
    # Weighting by the estimate (or by its projection) would let a region the
    # extractor happens to output loudly pull the pooled number toward its own
    # score -- reintroducing exactly the level sensitivity this function exists
    # to remove, one level up. The target is ground truth: how much real speech
    # sits at each depth is a property of the scene, not of the system.
    total_target = 0.0
    total_noise = 0.0
    scored = False
    for k in np.unique(depth):
        if k < min_depth:
            continue
        mask = depth == k
        target_energy = float(np.dot(target[mask], target[mask]))
        if target_energy < _EPS:
            continue  # silent target in this bucket: undefined, not zero-error
        ratio = 10.0 ** (si_sdr_regionwise(estimate, target, mask) / 10.0)
        scored = True
        total_target += target_energy
        total_noise += target_energy / ratio if ratio > 0.0 else float("inf")

    if not scored:
        return float("nan")
    if math.isinf(total_noise):
        return float("-inf")  # a region output silence against real speech
    if total_noise < _EPS:
        return float("inf")
    return 10.0 * np.log10(total_target / total_noise)


def depth_scale_factors(
    estimate: np.ndarray,
    target: np.ndarray,
    depth: np.ndarray,
    min_depth: int = 1,
) -> dict[int, float]:
    """The optimal SI-SDR scale ``alpha_k`` fitted in each depth region.

    Every SI-SDR in this project is scale-invariant, so a systematic level error
    in one region is invisible to all of them — it shows up only as a
    disagreement between the whole-track number and the per-depth ones, which is
    how it stayed unmeasured until 2026-08-23. Reporting the scales directly
    turns that inference into a measurement: ``max(alpha) / min(alpha)`` is the
    level disagreement across regions, and a globally rescaled estimate leaves
    that ratio unchanged (only cross-region disagreement moves it).

    Buckets whose target is silent are omitted, matching
    :func:`si_sdr_by_depth`.
    """
    estimate = np.asarray(estimate, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    depth = np.asarray(depth)

    scales: dict[int, float] = {}
    for k in np.unique(depth):
        if k < min_depth:
            continue
        mask = depth == k
        scale, _, _ = _region_energies(estimate[mask], target[mask])
        if not math.isnan(scale):
            scales[int(k)] = scale
    return scales


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


def si_sdr_by_depth(
    estimate: np.ndarray,
    target: np.ndarray,
    depth: np.ndarray,
) -> dict[int, float]:
    """SI-SDR stratified by overlap depth ``|K|`` (CLAUDE.md §5 Phase 2: "stratify
    every metric by overlap depth |K|" -- the evidence for the accumulation-free
    claim, not aggregate averages).

    ``depth`` (e.g. from :func:`dagger.diarize.oracle.overlap_depth`) is a
    per-sample concurrent-speaker count aligned with ``estimate``/``target``.
    Returns ``{k: si_sdr_regionwise(estimate, target, depth == k)}`` for every
    ``k >= 1`` present in ``depth`` (depth 0 -- nobody active -- isn't scoreable
    and is omitted).
    """
    depth = np.asarray(depth)
    return {
        int(k): si_sdr_regionwise(estimate, target, depth == k)
        for k in sorted(np.unique(depth))
        if k >= 1
    }


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
