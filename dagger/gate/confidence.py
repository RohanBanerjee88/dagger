"""The combined confidence gate (CLAUDE.md §2, §5 Phase 2).

Order matters: the ``V_i`` enrollment-variance check runs *first* and
short-circuits the rest -- "the gate can't check its own enrollment," so a
margin computed against a contaminated enrollment must never be allowed to
pass on its own. Only once enrollment is trusted do margin, VAD coverage, and
the artifact score each get an independent threshold; all must pass to accept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dagger.enroll.encoder import SpeakerEncoder
from dagger.gate.artifact import spectral_flatness, vad_coverage
from dagger.gate.enrollment import enrollment_variance_ok, mean_enrollment_variance
from dagger.gate.margin import identity_margin


@dataclass(frozen=True)
class GateDiagnostics:
    """The four measured quantities the gate thresholds are compared against.

    Deliberately threshold-free: these are properties of the estimate and its
    enrollment, not of any particular gate configuration. Separating them from
    the decision is what makes threshold selection cheap -- a candidate
    threshold set can be re-applied to recorded diagnostics via
    :func:`apply_thresholds` without re-running the encoder or the extractor
    (see ``scripts/tune_gate.py``).
    """

    mean_variance: float  # V_i, mean per-dimension enrollment variance
    margin: float  # M_i, identity margin (NOT raw similarity -- CLAUDE.md §2)
    vad_coverage: float
    artifact_score: float


@dataclass(frozen=True)
class GateResult:
    """One speaker's confidence-gate decision, plus the raw diagnostic values."""

    accepted: bool
    margin: float
    vad_coverage: float
    artifact_score: float
    reason: str  # "accepted", or the name of the first check that rejected it
    #: V_i, the value ``max_mean_variance`` is compared against. Recorded so a
    #: tuning pass can sweep that threshold from a dumped CSV -- without it,
    #: one of the four thresholds has no measured value anywhere in the run.
    #: nan when the diagnostics were short-circuited before it was needed.
    mean_variance: float = float("nan")


def gate_diagnostics(
    estimate: np.ndarray,
    sample_rate: int,
    embedding_self: np.ndarray,
    embeddings_others: list[np.ndarray],
    encoder: SpeakerEncoder,
    enrollment_variance: np.ndarray,
    expected_active: np.ndarray,
    *,
    precomputed_embedding: np.ndarray | None = None,
    artifact_min_energy_db: float | None = None
) -> GateDiagnostics:
    """Compute all four gate diagnostics unconditionally.

    :func:`confidence_gate` normally skips the last three whenever the ``V_i``
    check already rejected, which saves an encoder call per rejected speaker.
    That short-circuit is the right default for an eval run and the wrong one
    for a tuning run: it leaves ``margin``/``vad_coverage``/``artifact_score``
    undefined for exactly the speakers ``V_i`` killed, so those rows cannot
    contribute to a sweep over the other three thresholds. This function pays
    the extra encoder call to keep every row usable.
    """
    return GateDiagnostics(
        mean_variance=mean_enrollment_variance(enrollment_variance),
        margin=identity_margin(
            estimate, sample_rate, embedding_self, embeddings_others, encoder,
            precomputed_embedding=precomputed_embedding,
        ),
        vad_coverage=vad_coverage(estimate, expected_active, sample_rate),
        artifact_score=spectral_flatness(estimate, min_energy_db=artifact_min_energy_db)
    )


def apply_thresholds(
    diagnostics: GateDiagnostics,
    *,
    tau_margin: float,
    max_mean_variance: float,
    min_vad_coverage: float,
    max_artifact_score: float,
) -> GateResult:
    """Turn measured diagnostics into an accept/reject decision.

    Pure: four floats in, four thresholds in, a decision out -- no audio, no
    encoder, no model. That is what lets ``scripts/tune_gate.py`` evaluate
    thousands of candidate threshold sets against a recorded CSV instead of
    re-running the pipeline once per candidate.

    Check order is load-bearing and matches :func:`confidence_gate` exactly:
    ``V_i`` first (CLAUDE.md §2 -- "the gate can't check its own enrollment",
    so a margin computed against a contaminated enrollment must never pass on
    its own), then margin, coverage, artifact. ``reason`` names the FIRST check
    that failed, so it is a first-failure label, not a full failure set -- a
    later threshold never appearing there does not prove it is inert.

    NaN-safe by construction: a NaN diagnostic fails its ``>=``/``<=``
    comparison (NaN compares False against anything), so an undefined value
    rejects rather than silently passing.
    """
    if not (diagnostics.mean_variance <= max_mean_variance):
        return GateResult(
            accepted=False, margin=diagnostics.margin,
            vad_coverage=diagnostics.vad_coverage,
            artifact_score=diagnostics.artifact_score,
            reason="enrollment_variance", mean_variance=diagnostics.mean_variance,
        )
    if not (diagnostics.margin >= tau_margin):
        reason = "margin"
    elif not (diagnostics.vad_coverage >= min_vad_coverage):
        reason = "vad_coverage"
    elif not (diagnostics.artifact_score <= max_artifact_score):
        reason = "artifact_score"
    else:
        reason = "accepted"
    return GateResult(
        accepted=reason == "accepted", margin=diagnostics.margin,
        vad_coverage=diagnostics.vad_coverage,
        artifact_score=diagnostics.artifact_score,
        reason=reason, mean_variance=diagnostics.mean_variance,
    )


def confidence_gate(
    estimate: np.ndarray,
    sample_rate: int,
    embedding_self: np.ndarray,
    embeddings_others: list[np.ndarray],
    encoder: SpeakerEncoder,
    enrollment_variance: np.ndarray,
    expected_active: np.ndarray,
    *,
    tau_margin: float,
    max_mean_variance: float,
    min_vad_coverage: float,
    max_artifact_score: float,
    precomputed_embedding: np.ndarray | None = None,
    full_diagnostics: bool = False,
    artifact_min_energy_db: float | None = None
) -> GateResult:
    """Accept/reject one speaker's extracted estimate ``ŝ_i``.

    NaN-safe by construction: a NaN margin/coverage/artifact score fails its
    ``>=``/``<=`` comparison (NaN compares False against anything), so an
    undefined diagnostic (e.g. no other speakers to compare against) rejects
    rather than silently passing.

    ``precomputed_embedding``, if the caller already has
    ``encoder.embed(estimate, sample_rate)`` on hand, skips re-embedding
    ``estimate`` inside :func:`identity_margin` -- see that function's
    docstring for the exact contract.

    ``full_diagnostics=True`` computes margin/coverage/artifact even when the
    ``V_i`` check already rejected. The decision is identical either way; only
    the recorded numbers differ. Leave it off for eval runs (it costs one extra
    encoder call per V_i-rejected speaker) and turn it on for tuning runs,
    where a row with undefined diagnostics is a row that cannot contribute to a
    sweep over the other three thresholds.
    """
    thresholds = dict(
        tau_margin=tau_margin, max_mean_variance=max_mean_variance,
        min_vad_coverage=min_vad_coverage, max_artifact_score=max_artifact_score,
    )
    if not full_diagnostics and not enrollment_variance_ok(enrollment_variance, max_mean_variance):
        # Short-circuit: skip the three estimate-derived diagnostics (and the
        # encoder call inside identity_margin) for an enrollment already known
        # to be untrustworthy. apply_thresholds would reach the same verdict.
        return GateResult(
            accepted=False, margin=float("nan"), vad_coverage=float("nan"),
            artifact_score=float("nan"), reason="enrollment_variance",
            mean_variance=mean_enrollment_variance(enrollment_variance),
        )

    diagnostics = gate_diagnostics(
        estimate, sample_rate, embedding_self, embeddings_others, encoder,
        enrollment_variance, expected_active,
        precomputed_embedding=precomputed_embedding,
        artifact_min_energy_db=artifact_min_energy_db
    )
    return apply_thresholds(diagnostics, **thresholds)

def artifact_min_energy_db(gate_cfg: dict) -> float | None:
    """Read ``artifact_min_energy_db`` out of a ``gate:`` config block.

    Absent -> ``None`` -> the pre-2026-08-26 whole-track flatness, so every
    existing config is byte-identical. Present and ``null`` -> also ``None``,
    but *deliberately*, which is what ``scripts/tune_gate.py`` requires before
    it will recommend a ``max_artifact_score``.

    Exists as a named reader rather than four scattered ``.get`` calls because
    CLAUDE.md §9's first entry is a flag that was read, validated, warned about
    and never forwarded -- ~3.4 h of GPU producing output bit-identical to a
    plain run. One reader, four call sites, one test per call site.
    """
    return gate_cfg.get("artifact_min_energy_db")
