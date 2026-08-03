"""Confidence-gated embedding refinement (CLAUDE.md §1, §5 Phase 2: "coarse-to-fine").

Recursion here refines the *embedding* ``ē_i`` only -- it never touches the
audio path. Every round calls :func:`dagger.reconstruct.stitch.reconstruct_all`
unmodified, which always extracts from the untouched ``x_O`` (the same guarded
function Phase 1's ``no_recursion`` system uses); this module's only output is
a better embedding to feed it next round. This is what makes coarse-to-fine
accumulation-free where the deflation baselines
(:mod:`dagger.reconstruct.deflation`) are not: nothing here is ever subtracted
from the mixture to produce output audio.

Unlike the deflation baselines, processing *order* does not affect this
module's result: each round re-embeds every speaker from *this round's*
(already order-independent) reconstruction and only commits every accepted
update once the whole round is done, so speaker i's candidate never depends on
whether speaker j was refined before or after it in the same round.
"""

from __future__ import annotations

import numpy as np

from dagger.enroll.encoder import SpeakerEncoder
from dagger.extract.base import Extractor
from dagger.gate.confidence import GateResult, confidence_gate
from dagger.reconstruct.stitch import reconstruct_all


def _longest_run(mask: np.ndarray) -> tuple[int, int] | None:
    """The longest contiguous ``True`` run in ``mask``, as ``(start, end)``."""
    mask = np.asarray(mask).astype(bool)
    n = mask.shape[0]
    best: tuple[int, int] | None = None
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if best is None or (j - i) > (best[1] - best[0]):
                best = (i, j)
            i = j
        else:
            i += 1
    return best


def refine_embeddings(
    x,
    x_O,
    activity: np.ndarray,
    solo: np.ndarray,
    initial_embeddings: np.ndarray,
    enrollment_variances: np.ndarray,
    extractor: Extractor,
    encoder: SpeakerEncoder,
    sample_rate: int,
    *,
    rounds: int = 2,
    fade: int = 0,
    tau_margin: float,
    max_mean_variance: float,
    min_vad_coverage: float,
    max_artifact_score: float,
) -> tuple[np.ndarray, list[list[GateResult | None]]]:
    """Refine each speaker's embedding over ``rounds`` iterations.

    Each round: reconstruct every speaker with the current embeddings (via the
    unmodified, guarded ``reconstruct_all``), then for each speaker take the
    longest contiguous overlap-region run of their *own* reconstruction (a
    purer sample of that speaker than the raw mixture, now that extraction has
    separated them out) and re-embed it. The candidate embedding is a running
    mean of the previous embedding and this new estimate (never a full
    replacement -- one bad round can't discard a good enrollment), gated by
    :func:`dagger.gate.confidence.confidence_gate` before being accepted. A
    rejected candidate leaves that speaker's embedding unchanged for the next
    round.

    The gate judges the re-embedded clip against the speaker's CURRENT
    embedding, never against the blended candidate -- see the comment at the
    call site for why a blended reference silently defeats the margin. Note the
    reference does move across rounds (an accepted update changes it), so with
    many rounds the anchor could drift along with the thing it is judging; at
    the default ``rounds=2`` an accepted embedding stays within 3/4 of the way
    from the original enrollment, which bounds that. If drift ever looks like a
    problem, anchoring on ``initial_embeddings[i]`` instead is the one-line
    alternative -- it trades drift-immunity for an increasingly stale reference.

    Returns ``(final_embeddings, round_results)`` -- the caller (e.g.
    ``scripts/run_phase2.py``) makes one final ``reconstruct_all`` call with
    ``final_embeddings`` to get the actual output audio; this function never
    produces audio itself. ``round_results[r][i]`` is speaker ``i``'s gate
    decision in round ``r`` (every round is kept, not just the last, so a
    caller can tell "the gate rubber-stamped everything from round 0" apart
    from "refinement degraded progressively"), or ``None`` when that speaker
    had no overlap-only region to re-embed from that round.
    """
    num_speakers = activity.shape[0]
    embeddings = np.array(initial_embeddings, dtype=np.float64, copy=True)
    round_results: list[list[GateResult | None]] = []

    for _round in range(rounds):
        outputs = reconstruct_all(x, x_O, activity, solo, embeddings, extractor, fade=fade)
        candidate_embeddings = embeddings.copy()
        round_results.append([None] * num_speakers)

        for i in range(num_speakers):
            overlap_i = (activity[i] > 0) & (solo[i] <= 0)
            run = _longest_run(overlap_i)
            if run is None:
                continue
            start, end = run
            clip = outputs[i][start:end]

            raw_embedding = encoder.embed(clip, sample_rate)
            blended = 0.5 * embeddings[i] + 0.5 * raw_embedding
            others = [embeddings[j] for j in range(num_speakers) if j != i]

            result = confidence_gate(
                clip,
                sample_rate,
                # The CURRENT embedding, not `blended`. Passing the blend made
                # the margin self-referential: identity_margin embeds `clip`
                # (that is `raw_embedding`) and compares it against
                # `embedding_self`, so a blend containing 0.5*raw_embedding is
                # half-made of the very thing being judged. For a candidate at
                # angle theta from the enrollment that replaces cos(theta) with
                # cos(theta/2) -- inflating the "same" term for every candidate,
                # and inflating it MOST for the worst ones (at 90 degrees,
                # 0.00 becomes 0.71), which is backwards for a gate. Symptom in
                # the 2026-08-02 Phase 2 runs: a 98-99% accept rate that did not
                # move with speaker count, while the identically-thresholded
                # gate in dagger.reconstruct.deflation tracked difficulty
                # correctly (71.8% -> 61.2% -> 54.1% at m=3/4/5).
                # This restores CLAUDE.md §2's definition,
                # M_i = cos(s_hat_i, e_i) - max_{j!=i} cos(s_hat_i, e_j), and
                # matches how scripts/run_phase2.py's deflation gate already
                # calls it.
                embeddings[i],
                others,
                encoder,
                enrollment_variances[i],
                np.ones(end - start, dtype=bool),
                tau_margin=tau_margin,
                max_mean_variance=max_mean_variance,
                min_vad_coverage=min_vad_coverage,
                max_artifact_score=max_artifact_score,
                # raw_embedding == encoder.embed(clip, sample_rate), computed
                # just above -- avoids a redundant TitaNet call inside
                # confidence_gate's margin check for the same clip.
                precomputed_embedding=raw_embedding,
            )
            round_results[-1][i] = result
            if result.accepted:
                candidate_embeddings[i] = blended

        embeddings = candidate_embeddings

    return embeddings, round_results
