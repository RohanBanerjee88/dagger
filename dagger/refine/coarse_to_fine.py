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

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from dagger.enroll.encoder import SpeakerEncoder
from dagger.extract.base import Extractor
from dagger.gate.confidence import GateResult, confidence_gate
from dagger.reconstruct.stitch import reconstruct_all

#: ``reason`` values used only when an ``accept_fn`` overrides the gate. They
#: record the 2x2 of (what the alternative rule did, what the gate would have
#: done) in the existing field, so no CSV schema anywhere has to change -- and
#: the two disagreement rows are the whole point of a ceiling run:
#: ``ceiling_accept_gate_would_reject`` is headroom the real gate cannot reach.
CEILING_ACCEPT_GATE_REJECT = "ceiling_accept_gate_would_reject"
CEILING_REJECT_GATE_ACCEPT = "ceiling_reject_gate_would_accept"


def _override_reason(committed: bool, gate: GateResult) -> str:
    """How an overridden decision is labelled, preserving the gate's verdict."""
    if committed and gate.accepted:
        return "accepted"
    if committed:
        return CEILING_ACCEPT_GATE_REJECT
    if gate.accepted:
        return CEILING_REJECT_GATE_ACCEPT
    return gate.reason


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
    min_clip_ms: float = 50.0,
    tau_margin: float,
    max_mean_variance: float,
    min_vad_coverage: float,
    max_artifact_score: float,
    accept_fn: Callable[[int, np.ndarray, np.ndarray], bool] | None = None,
    candidate_audio: np.ndarray | None = None,
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

    ``min_clip_ms`` is the shortest overlap run that will be re-embedded. It is
    a floor for what the speaker encoder can physically process (below about one
    mel frame TitaNet raises), NOT the 500 ms *stability* threshold enrollment
    applies. Runs below it yield a rejected ``GateResult`` with reason
    ``"overlap_clip_too_short"``, distinct from the ``None`` used for "no
    overlap-only region at all" -- under real diarization those two are
    different failures and the difference is informative.

    ``accept_fn`` replaces the gate's *commit* decision with an arbitrary rule,
    called as ``accept_fn(i, candidate_output, current_output)`` on this round's
    reconstructed audio. ``None`` -- the default and the only deployable setting
    -- keeps the confidence gate, and the code path is then bit-identical to
    before this argument existed.

    It exists for ONE experiment: the oracle-refinement **ceiling** (CLAUDE.md
    §5 Phase 3, outstanding item 4). Refinement is now measured net-negative in
    every regime tried -- clean enrollment, heterogeneous enrollment, and
    contaminated real-diarization enrollment -- but "never positive" cannot be
    established by accumulating negatives. Substituting a rule that can see the
    ground truth answers the different, decidable question: does refinement have
    any headroom on this extractor at all? A still-negative ceiling is a
    publishable negative result with a stated mechanism; a positive ceiling the
    real gate cannot find means the acceptance RULE is what is broken, not
    refinement. See :mod:`dagger.refine.oracle_ceiling`.

    ``candidate_audio`` (``[S, T]``, default ``None``) replaces the audio the
    candidate embedding is computed FROM. ``None`` -- the default and the only
    deployable setting -- re-embeds this round's own reconstruction, and the
    code path is then bit-identical to before this argument existed.

    It exists for the other half of the refinement bracket. ``accept_fn`` bounds
    the *acceptance rule*; this bounds the *extractor*. Refinement blends
    ``0.5*e_enrolled + 0.5*e_from_extracted_overlap``, so for it to pay, the
    second term must beat the first -- and it is embedded from ``G``'s output,
    which currently sits near 1-2 dB and is therefore dominated by artifacts.
    Every "refinement is net-harmful" result this project has (clean, starved,
    heterogeneous, and contaminated real-diarization enrollment) varied
    ENROLLMENT quality while holding extractor quality fixed at that one poor
    operating point. Passing the clean sources here is the perfect-extractor
    limit: if refinement still loses, its premise is wrong regardless of how
    good ``G`` ever gets, and the idea can be closed out. If it wins, the
    working region exists and its boundary lies between today's ``G`` and
    perfect -- which makes refinement a question about the training budget, not
    about the update rule.

    Like ``accept_fn`` it reads ground truth and is **NOT DEPLOYABLE**; it is a
    scoring-time bound only.

    The gate still runs and its verdict is still recorded when ``accept_fn`` is
    given -- ``GateResult.accepted`` reports what was actually committed, and
    ``reason`` records where the two disagreed
    (``ceiling_accept_gate_would_reject`` is precisely the headroom the real
    gate missed). Cost is one extra ``reconstruct_all`` per round: each
    speaker's output depends only on its own embedding, so every candidate is
    evaluated in a single batched call rather than one per speaker.
    """
    num_speakers = activity.shape[0]
    embeddings = np.array(initial_embeddings, dtype=np.float64, copy=True)
    round_results: list[list[GateResult | None]] = []
    # A technical floor for the ENCODER, not a quality threshold: below roughly
    # one mel frame the speaker encoder cannot produce an embedding at all. The
    # 500 ms stability floor that enrollment uses (`select_topk_solo_clips`) is
    # a different, stricter judgement and is deliberately not reused here --
    # raising this value silently changes which speakers get refined at all.
    min_samples = max(1, int(round(min_clip_ms / 1000.0 * sample_rate)))

    for _round in range(rounds):
        outputs = reconstruct_all(x, x_O, activity, solo, embeddings, extractor, fade=fade)
        candidate_embeddings = embeddings.copy()
        round_results.append([None] * num_speakers)
        # (speaker -> blended candidate) for the speakers that produced one. The
        # commit decision is deferred to a second pass so an `accept_fn` can
        # judge every candidate's reconstructed AUDIO, which needs all of them
        # in hand. Order-independence is unaffected: `others` and the blend
        # below still read `embeddings`, this round's start-of-round values.
        proposals: dict[int, np.ndarray] = {}

        for i in range(num_speakers):
            overlap_i = (activity[i] > 0) & (solo[i] <= 0)
            run = _longest_run(overlap_i)
            if run is None:
                continue
            start, end = run
            if (end - start) < min_samples:
                # Too short to re-embed. Under ORACLE diarization this never
                # fires -- the scene scheduler gives every speaker a long
                # overlap tail -- but a real diarizer's boundaries are jittery,
                # and an overlap-only run of a few samples is routine. TitaNet
                # then gets fewer samples than one mel frame and NeMo raises
                # ("normalize_batch ... received a tensor of length 1"), which
                # took down a whole Phase 3 run at scene 9 of 150.
                #
                # Recorded with its OWN reason rather than reusing
                # "no_overlap_clip": "this speaker never overlapped anyone" and
                # "this speaker's overlap was too brief to use" are different
                # facts about the diarization, and collapsing them would hide
                # how much refinement real boundary jitter actually costs.
                round_results[-1][i] = GateResult(
                    accepted=False,
                    margin=float("nan"),
                    vad_coverage=float("nan"),
                    artifact_score=float("nan"),
                    reason="overlap_clip_too_short",
                    mean_variance=float("nan"),
                )
                continue
            # Under `candidate_audio` the gate judges the SAME clip the
            # candidate was embedded from, not the deployable pipeline's output.
            # That is deliberate and it is what makes the counterfactual
            # coherent: the question is "what would refinement be worth if `G`
            # were perfect", and in that world the gate sees the good audio too.
            # Judging clean-audio candidates with a gate looking at ~2 dB output
            # was tried first and rejected 100% of them -- the bound then
            # measures the gate rather than the extractor, which is the axis
            # `accept_fn` already covers.
            clip = outputs[i][start:end] if candidate_audio is None else candidate_audio[i][start:end]

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
            proposals[i] = blended

        if accept_fn is None:
            for i, blended in proposals.items():
                if round_results[-1][i].accepted:
                    candidate_embeddings[i] = blended
        elif proposals:
            # One batched reconstruction with every candidate applied at once.
            # Legal because reconstruct_all extracts each speaker from the same
            # untouched x_O using only that speaker's own embedding (CLAUDE.md
            # §1) -- so speaker i's row here is exactly what it would be if only
            # i's candidate had been applied. That is what makes a single call
            # correct rather than an approximation.
            trial = embeddings.copy()
            for i, blended in proposals.items():
                trial[i] = blended
            trial_outputs = reconstruct_all(
                x, x_O, activity, solo, trial, extractor, fade=fade
            )
            for i, blended in proposals.items():
                gate_result = round_results[-1][i]
                committed = bool(accept_fn(i, trial_outputs[i], outputs[i]))
                round_results[-1][i] = replace(
                    gate_result,
                    accepted=committed,
                    reason=_override_reason(committed, gate_result),
                )
                if committed:
                    candidate_embeddings[i] = blended

        embeddings = candidate_embeddings

    return embeddings, round_results
