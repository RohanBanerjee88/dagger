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
from dagger.gate.enrollment import enrollment_variance_ok
from dagger.gate.margin import identity_margin


@dataclass(frozen=True)
class GateResult:
    """One speaker's confidence-gate decision, plus the raw diagnostic values."""

    accepted: bool
    margin: float
    vad_coverage: float
    artifact_score: float
    reason: str  # "accepted", or the name of the first check that rejected it


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
) -> GateResult:
    """Accept/reject one speaker's extracted estimate ``ŝ_i``.

    NaN-safe by construction: a NaN margin/coverage/artifact score fails its
    ``>=``/``<=`` comparison (NaN compares False against anything), so an
    undefined diagnostic (e.g. no other speakers to compare against) rejects
    rather than silently passing.
    """
    if not enrollment_variance_ok(enrollment_variance, max_mean_variance):
        return GateResult(
            accepted=False, margin=float("nan"), vad_coverage=float("nan"),
            artifact_score=float("nan"), reason="enrollment_variance",
        )

    margin = identity_margin(estimate, sample_rate, embedding_self, embeddings_others, encoder)
    coverage = vad_coverage(estimate, expected_active, sample_rate)
    artifact = spectral_flatness(estimate)

    if not (margin >= tau_margin):
        return GateResult(False, margin, coverage, artifact, "margin")
    if not (coverage >= min_vad_coverage):
        return GateResult(False, margin, coverage, artifact, "vad_coverage")
    if not (artifact <= max_artifact_score):
        return GateResult(False, margin, coverage, artifact, "artifact_score")
    return GateResult(True, margin, coverage, artifact, "accepted")
