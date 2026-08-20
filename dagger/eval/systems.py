"""The four Phase 2 systems, scored on one scene.

Moved here from ``scripts/run_phase2.py`` when Phase 3 needed to run the same
comparison twice per scene (oracle regions vs. real ones). ``run_phase2.py``
re-exports every public name, so its behaviour and its module-level API are
unchanged — the guard on that claim is that it must still reproduce
``results/phase2/dod_final/`` byte-identically (the eval path has no RNG
anywhere, confirmed 2026-07-29 by reproducing a table to the decimal).

The systems, all conditioning the SAME trained extractor ``G`` and differing only
in how they reconstruct (CLAUDE.md §5 Phase 2):

* ``no_recursion``       — Phase 1's proposed path: accumulation-free, no gate,
                           no refinement. The control.
* ``ungated_deflation``  — the deliberate anti-pattern (CLAUDE.md §1): subtract
                           each estimate into a running residual, re-extract from
                           it. Built only to be compared against.
* ``gated_deflation``    — same, but a rejected estimate leaves the residual
                           untouched.
* ``coarse_to_fine``     — "ours": recursion refines the *embedding* only; audio
                           always comes from the unmodified, guarded ``x_O``.

The one structural change made during the move is :func:`deflation_order` — see
its docstring, which is a Phase 3 correctness matter, not a tidy-up.
"""

from __future__ import annotations

import numpy as np

from dagger.audio.provenance import original_mixture
from dagger.diarize.oracle import overlap_depth, overlap_mixture
from dagger.enroll.topk import NoSoloRegionError, enroll_speaker
from dagger.gate.confidence import confidence_gate
from dagger.metrics.sisdr import si_sdr, si_sdr_by_depth
from dagger.reconstruct.deflation import reconstruct_all_deflation
from dagger.reconstruct.stitch import reconstruct_all
from dagger.refine.coarse_to_fine import refine_embeddings

SYSTEMS = ("no_recursion", "ungated_deflation", "gated_deflation", "coarse_to_fine")

# The two systems that run dagger.reconstruct.deflation, i.e. the only ones for
# which `deflation_index`/`n_accepted_before` mean anything. Membership is
# explicit rather than a name test (`endswith("deflation")`) so adding a system
# can't silently opt it in.
DEFLATION_SYSTEMS = ("ungated_deflation", "gated_deflation")

SCORE_FIELDS = [
    "scene", "speaker", "system", "m", "depth", "si_sdr",
    "deflation_index", "n_accepted_before", "refine_rounds",
]
# The un-stratified, whole-output number (CLAUDE.md §7). Its own file at its own
# grain -- one row per (scene, speaker, system), NOT per depth -- for the same
# reason gate decisions live apart: a depth-stratified table that silently
# absorbed a whole-output row would report a mean over a pipeline, labelled as a
# depth. This project has shipped that mistake three times (the +-inf drop, the
# `dilate_ms` sweep, the ceiling's objective), so the grain is enforced by file
# separation rather than by remembering to filter.
#
# What it answers: "is this configuration net better?" -- unanswerable from
# per-depth rows alone, because a gain and its cost land in different rows with
# no exchange rate between them. Stage B's dilation sweep is the worked case:
# +2.19 dB at depth 2 against -29.85 dB at depth 1, with no way to net them.
#
# CAUTION, and it is not optional. This number is SCALE-ANCHORED by the
# bit-exact solo copy that dominates most outputs, so it rewards fixing a level
# error over fixing a shape error. It is a REPORTING number only -- never a
# selection, gating or optimization target. Using it as one is exactly what
# voided Stage B's refinement ceiling (see dagger/refine/oracle_ceiling.py).
OVERALL_FIELDS = [
    "scene", "speaker", "system", "m", "si_sdr",
    "deflation_index", "n_accepted_before", "refine_rounds",
]
# Gate decisions live in their own file, at their own grain: one per (scene,
# speaker, system, round). Folding them into the score rows would duplicate each
# decision across however many depths that speaker happens to span, so any
# accept rate counted from those rows would be inflated by a depth-dependent
# factor. Keeping the two grains in separate files makes that mistake hard.
GATE_FIELDS = [
    "scene", "speaker", "system", "round",
    "accepted", "mean_variance", "margin", "vad_coverage", "artifact_score", "reason",
]

#: Valid values for ``deflation.order``. See :func:`deflation_order`.
ORDER_POLICIES = ("variance", "index")


def deflation_order(variances: np.ndarray, policy: str = "variance") -> list[int]:
    """The order deflation processes speakers in.

    ``"variance"`` — ascending mean ``V_i``, i.e. the most-confidently-enrolled
    speaker is deflated first. This is the historical behaviour and stays the
    default.

    ``"index"`` — plain ``[0, 1, ..., m-1]``.

    **Why this is a named policy rather than an inline ``sorted``.** Under oracle
    diarization ``V_i`` is *identically zero* for every speaker (the scene
    scheduler gives each exactly one solo run, so enrollment yields one clip and
    the variance across one sample is 0). Python's ``sorted`` is stable, so
    ``"variance"`` has always returned index order and the sort has never
    actually done anything.

    Real diarization changes that: it fragments solo regions, enrollment draws
    several clips, and ``V_i`` becomes a real, data-dependent quantity. The sort
    then becomes a real permutation — and position in this order *is*
    ``n_accepted_before``, the dominant driver of SI-SDR for the deflation
    systems (Phase 2's headline: ~1.8 dB across levels).

    So an oracle-vs-real comparison of a deflation system would vary two things
    at once, and not symmetrically: whichever speaker's solo region best survives
    diarization gets the lowest ``V_i`` and is promoted to position 0, the
    least-damaged accumulation slot. That speaker's extraction was already the
    easiest, so the real arm is *flattered* and the measured gap is biased
    **downwards** in a predictable direction — which a caveat cannot repair.
    Forcing ``"index"`` in a parallel arm separates the two effects.
    """
    if policy not in ORDER_POLICIES:
        raise ValueError(
            f"deflation.order must be one of {ORDER_POLICIES}, got {policy!r}."
        )
    num_speakers = int(np.asarray(variances).shape[0])
    if policy == "index":
        return list(range(num_speakers))
    return sorted(range(num_speakers), key=lambda i: float(np.mean(variances[i])))


def make_gate_fn(embeddings, variances, activity, overlap, encoder, sample_rate, gate_cfg):
    num_speakers = embeddings.shape[0]

    def gate_fn(i: int, estimate: np.ndarray):
        others = [embeddings[j] for j in range(num_speakers) if j != i]
        expected_active = activity[i].astype(bool) & overlap.astype(bool)
        return confidence_gate(
            estimate, sample_rate, embeddings[i], others, encoder,
            variances[i], expected_active,
            tau_margin=gate_cfg["tau_margin"],
            max_mean_variance=gate_cfg["max_mean_variance"],
            min_vad_coverage=gate_cfg["min_vad_coverage"],
            max_artifact_score=gate_cfg["max_artifact_score"],
        )

    return gate_fn


def accepted_before(order: list[int], accept_sequence: np.ndarray) -> list[int]:
    """Speaker-indexed count of accepted estimates subtracted into the residual
    *before* each speaker was extracted -- the accumulation counter Theorem 3's
    ``L*||E_(m-1)||`` penalty is indexed by.

    ``accept_sequence`` is speaker-indexed (see
    :func:`dagger.reconstruct.deflation.reconstruct_all_deflation`), so the
    prefix has to be taken over ``order``, not over speaker ids. Ungated
    deflation accepts unconditionally, so there this must equal each speaker's
    position in ``order``; gated deflation can only be lower.
    """
    return [
        sum(int(accept_sequence[j]) for j in order[: order.index(i)])
        for i in range(len(order))
    ]


def score_scene(
    scene,
    fade: int,
    enroll_k: int,
    min_clip_ms: float,
    enroll_budget_ms: float | None,
    encoder,
    extractor,
    gate_cfg: dict,
    refine_rounds: int,
    regions=None,
    order_policy: str = "variance",
    on_unenrollable: str = "raise",
    refine_oracle_ceiling: bool = False,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Run all four systems on one scene.

    Returns ``(score_rows, gate_rows)`` -- one score row per (speaker, depth,
    system), and one gate row per (speaker, system, round) for the two systems
    that actually run the confidence gate. See ``SCORE_FIELDS``/``GATE_FIELDS``
    for why the two are kept apart.

    ``regions`` is the who-spoke-when input (a
    :class:`~dagger.diarize.regions.Regions`). ``None`` means oracle diarization
    derived from the scene itself, which is exactly what Phase 2 did and is why
    ``run_phase2.py`` can call this without knowing the parameter exists. Phase 3
    passes real regions here — and note the audio path then runs entirely on
    *predicted* rows, with no knowledge of the true labels. That is deliberate:
    at inference on real data there are no true labels, so anything else would be
    leakage. Row attribution happens later, at scoring time.

    ``order_policy`` selects the deflation order; see :func:`deflation_order`.

    ``on_unenrollable`` decides what happens when a row has no usable solo audio:

    * ``"raise"`` (default) — propagate :class:`~dagger.enroll.topk.NoSoloRegionError`
      so the caller skips the whole scene. Phase 2's behaviour, unchanged.
    * ``"drop"`` — drop that row and score the rest. Phase 3 needs this because a
      real diarizer routinely invents a cluster that lies entirely inside another
      speaker's speech, and such a cluster can never be enrolled. Failing the
      whole scene on that would discard scenes in proportion to how badly the
      diarizer did — precisely the selection bias that silently shrank Phase 1's
      effective dataset (CLAUDE.md's `stagger_offsets` note). An unenrollable
      speaker is a real outcome and is reported, not hidden.

    ``refine_oracle_ceiling`` replaces refinement's confidence gate with a rule
    that reads the clean sources — see :mod:`dagger.refine.oracle_ceiling`. It is
    **NOT DEPLOYABLE** and exists to bound a component this project is otherwise
    about to report as net-harmful in every regime tested. Only
    ``coarse_to_fine`` is affected; the other three systems never refine, so
    their rows are unchanged and stay a valid control within the same run.
    """
    if regions is None:
        from dagger.diarize.oracle import OracleDiarizer
        from dagger.diarize.regions import scene_regions

        regions = scene_regions(scene, OracleDiarizer())

    if on_unenrollable not in ("raise", "drop"):
        raise ValueError(
            f"on_unenrollable must be 'raise' or 'drop', got {on_unenrollable!r}."
        )

    activity, speakers = regions.activity, regions.speakers
    solo, overlap = regions.solo, regions.overlap

    x = original_mixture(scene.mixture, label="x")
    x_O = overlap_mixture(x, overlap, label="x_O")

    # Ground truth enters here, and ONLY for scoring. Two uses, both strictly
    # after every waveform is produced, neither reaching the audio path:
    #
    #  * `targets`        -- which clean source each output row is compared
    #                        against (identity under oracle regions; the optimal
    #                        cluster mapping under real ones).
    #  * `scoring_depth`  -- which bucket each sample's score is reported in.
    #
    # `scoring_depth` comes from the REFERENCE, never from `regions.depth`.
    # Depth is defined as intrinsic difficulty (CLAUDE.md §6.4) -- a property of
    # the acoustic scene, not of the diarizer's opinion about it. Bucketing by
    # predicted depth lets a diarizer grade itself on a curve it drew: merge
    # three concurrent speakers into one cluster and every sample of a genuine
    # 3-way overlap gets filed under "depth 1", the easy row. It also breaks the
    # cross-arm pairing, since `depth 2` would then name a different set of
    # samples in each arm and their difference would not be a controlled
    # comparison. Same principle as DER's denominator, which is reference
    # speech (`dagger.metrics.der`), not predicted speech.
    #
    # This does NOT hand the method the speaker count. The audio path below runs
    # entirely on `activity`/`solo`/`overlap` from the diarizer, and the number
    # of output tracks is still whatever it found.
    reference_activity = _reference_activity(scene)
    scoring_depth = overlap_depth(reference_activity)
    targets = _score_targets(scene, regions, reference_activity)

    n_clusters = int(activity.shape[0])  # before any enrollment drop
    kept: list[int] = []
    enrollments = []
    for i in range(len(speakers)):
        try:
            enrollments.append(
                enroll_speaker(
                    scene.mixture, solo[i], activity[i], scene.sample_rate,
                    encoder, k=enroll_k, min_clip_ms=min_clip_ms,
                    budget_ms=enroll_budget_ms,
                )
            )
        except NoSoloRegionError:
            if on_unenrollable == "raise":
                raise
            continue
        kept.append(i)

    if not kept:
        raise NoSoloRegionError(
            f"score_scene: no row in scene {scene.name!r} had usable solo audio."
        )

    # Restrict every per-speaker array to the enrollable rows, together, so row
    # `j` means the same speaker in all of them.
    activity = activity[kept]
    solo = solo[kept]
    speakers = [speakers[i] for i in kept]
    targets = [targets[i] for i in kept]
    num_speakers = len(kept)

    embeddings = np.stack([e.embedding for e in enrollments], axis=0)
    variances = np.stack([e.variance for e in enrollments], axis=0)

    # Order is load-bearing only for the deflation systems -- coarse_to_fine's
    # batched refinement is order-independent, see dagger.refine.coarse_to_fine.
    order = deflation_order(variances, order_policy)
    deflation_idx: dict[int, int] = {i: j for j, i in enumerate(order)}  # speaker -> position in order

    outputs: dict[str, np.ndarray] = {}
    outputs["no_recursion"] = reconstruct_all(x, x_O, activity, solo, embeddings, extractor, fade=fade)

    # Each deflation call binds its OWN result names: reusing one pair across
    # both calls would let the gated run's telemetry overwrite the ungated
    # run's and end up reported against the wrong system.
    outputs["ungated_deflation"], _, ungated_accepts = reconstruct_all_deflation(
        x, x_O, activity, solo, embeddings, extractor, order, gate_fn=None, fade=fade,
    )
    gate_fn = make_gate_fn(embeddings, variances, activity, overlap, encoder, scene.sample_rate, gate_cfg)
    outputs["gated_deflation"], gated_gate_results, gated_accepts = reconstruct_all_deflation(
        x, x_O, activity, solo, embeddings, extractor, order, gate_fn=gate_fn, fade=fade,
    )

    n_accepted_before = {
        "ungated_deflation": accepted_before(order, ungated_accepts),
        "gated_deflation": accepted_before(order, gated_accepts),
    }
    # Ungated accepts unconditionally, so its accumulation count must be exactly
    # each speaker's position in `order`. Deriving it from the returned accept
    # sequence rather than hardcoding that lets this assertion catch a
    # regression in the deflation loop itself.
    assert n_accepted_before["ungated_deflation"] == [deflation_idx[i] for i in range(num_speakers)]

    refine_accept_fn = None
    if refine_oracle_ceiling:
        from dagger.refine.oracle_ceiling import make_oracle_accept_fn

        # Built from the ALREADY-RESTRICTED targets, so row i here is the same
        # row i the refiner sees. Ground truth reaches only the accept/reject
        # decision; the audio is still G(x_O, e) per CLAUDE.md §1.
        #
        # `scoring_depth` is passed so the rule optimizes the SAME slice the
        # results table reports. Without it the ceiling scored the whole
        # waveform, whose SI-SDR is dominated by the bit-exact solo copy -- a
        # different objective, under which the bound is not a bound. It came in
        # BELOW no_recursion on 2026-08-19 for exactly that reason.
        refine_accept_fn = make_oracle_accept_fn(
            [target for _, target in targets], scoring_depth
        )

    refined_embeddings, round_results = refine_embeddings(
        x, x_O, activity, solo, embeddings, variances, extractor, encoder, scene.sample_rate,
        rounds=refine_rounds, fade=fade,
        tau_margin=gate_cfg["tau_margin"], max_mean_variance=gate_cfg["max_mean_variance"],
        min_vad_coverage=gate_cfg["min_vad_coverage"], max_artifact_score=gate_cfg["max_artifact_score"],
        accept_fn=refine_accept_fn,
    )
    outputs["coarse_to_fine"] = reconstruct_all(
        x, x_O, activity, solo, refined_embeddings, extractor, fade=fade
    )

    rows = []
    overall_rows = []
    for system_name in SYSTEMS:
        system_outputs = outputs[system_name]
        is_deflation = system_name in DEFLATION_SYSTEMS
        for i, cluster in enumerate(speakers):
            label, target = targets[i]
            if target is None:
                # A predicted cluster with no ground-truth counterpart: there is
                # nothing to score it against. Reported as a spurious cluster in
                # the diarization diagnostics, not silently dropped.
                continue
            # The un-stratified number, scored over the WHOLE output track
            # against the whole true source. Outside a speaker's activity the
            # reconstruction is exactly 0 by construction (w_Ei + w_Oi ==
            # activity_i), so this pools every depth for that speaker without
            # also scoring silence the pipeline never claimed.
            overall_rows.append({
                "scene": scene.name,
                "speaker": label,
                "cluster": cluster,
                "n_clusters": n_clusters,
                "system": system_name,
                "m": num_speakers,
                "si_sdr": si_sdr(system_outputs[i], target),
                "deflation_index": deflation_idx[i] if is_deflation else None,
                "n_accepted_before": n_accepted_before[system_name][i] if is_deflation else None,
                "refine_rounds": refine_rounds if system_name == "coarse_to_fine" else 0,
            })

            by_depth = si_sdr_by_depth(system_outputs[i], target, scoring_depth)
            for k, score in by_depth.items():
                rows.append({
                    "scene": scene.name,
                    # The GROUND-TRUTH speaker this row was attributed to, not
                    # the cluster id. Paired comparisons key on (scene, speaker,
                    # depth, system), and a real diarizer's rows are labelled
                    # SPEAKER_00.. while the oracle's are s1.. -- keying on the
                    # raw cluster id would make every oracle-vs-real pair miss
                    # and silently produce an EMPTY gap table.
                    "speaker": label,
                    # Provenance, Phase 3 only. run_phase2.py's writer ignores
                    # extra keys, so the Phase 2 CSV schema is untouched.
                    "cluster": cluster,
                    # How many clusters the diarizer produced, BEFORE the
                    # enrollment drop. `m` is the count that survived, so the
                    # pair tells the whole story: n_clusters=2 with m=1 means
                    # one cluster had no solo region and could not be enrolled.
                    # Reading that off the 2026-08-16 run took a forensic pass
                    # over the CSV; it belongs in the table.
                    "n_clusters": n_clusters,
                    "system": system_name,
                    "m": num_speakers,
                    "depth": k,
                    "si_sdr": score,
                    # None (not 0) for the accumulation-free systems: 0 is a
                    # legitimate deflation_index (the first speaker in `order`),
                    # so writing it here would make "no accumulation" and "not
                    # applicable" indistinguishable in the CSV.
                    "deflation_index": deflation_idx[i] if is_deflation else None,
                    "n_accepted_before": n_accepted_before[system_name][i] if is_deflation else None,
                    "refine_rounds": refine_rounds if system_name == "coarse_to_fine" else 0,
                })

    gate_rows = []
    for i, cluster in enumerate(speakers):
        result = gated_gate_results[i]
        if result is not None:
            gate_rows.append({
                "scene": scene.name, "speaker": targets[i][0], "cluster": cluster,
                "system": "gated_deflation", "round": 0,
                "accepted": result.accepted, "mean_variance": result.mean_variance,
                "margin": result.margin, "vad_coverage": result.vad_coverage,
                "artifact_score": result.artifact_score, "reason": result.reason,
            })
    for round_index, per_speaker in enumerate(round_results):
        for i, cluster in enumerate(speakers):
            result = per_speaker[i]
            row = {
                "scene": scene.name, "speaker": targets[i][0], "cluster": cluster,
                "system": "coarse_to_fine", "round": round_index,
            }
            if result is None:
                # No overlap-only region to re-embed from, so no gate decision
                # was made. Recorded rather than skipped -- a missing row would
                # be silently indistinguishable from a rejection when accept
                # rates are counted.
                row.update({
                    "accepted": None, "mean_variance": None, "margin": None,
                    "vad_coverage": None, "artifact_score": None,
                    "reason": "no_overlap_clip",
                })
            else:
                row.update({
                    "accepted": result.accepted, "mean_variance": result.mean_variance,
                    "margin": result.margin, "vad_coverage": result.vad_coverage,
                    "artifact_score": result.artifact_score, "reason": result.reason,
                })
            gate_rows.append(row)

    return rows, gate_rows, overall_rows


def _score_targets(
    scene, regions, reference_activity=None
) -> list[tuple[str | None, np.ndarray | None]]:
    """``(ground-truth label, source)`` for each *output* row, ``(None, None)`` if
    the row cannot be attributed to any ground-truth speaker.

    Under oracle diarization row ``i`` is source ``i`` and this is the identity.
    Under real diarization output rows are anonymous clusters, so they are
    matched to sources by :mod:`dagger.diarize.mapping`.

    This is the ONLY place ground truth meets a predicted row, and it happens
    strictly after every waveform has been produced — the extractor, the gate,
    the enrollment and the deflation order all ran on predicted rows alone. At
    inference on real data there are no true labels, so anything else would be
    leakage rather than evaluation.
    """
    num_rows = regions.activity.shape[0]
    if list(regions.speakers) == list(scene.speakers):
        return [(scene.speakers[i], scene.sources[i]) for i in range(num_rows)]

    from dagger.diarize.mapping import map_clusters_to_speakers

    if reference_activity is None:
        reference_activity = _reference_activity(scene)
    mapping = map_clusters_to_speakers(regions.activity, reference_activity)
    return [
        (None, None) if ref is None else (scene.speakers[ref], scene.sources[ref])
        for ref in mapping.cluster_to_ref
    ]


def _reference_activity(scene) -> np.ndarray:
    """Oracle activity for ``scene``, used only to attribute predicted rows."""
    from dagger.diarize.oracle import OracleDiarizer
    from dagger.diarize.regions import scene_regions

    return scene_regions(scene, OracleDiarizer()).activity
