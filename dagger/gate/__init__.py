"""Confidence gate: identity margin ``M_i``, VAD, artifact score, threshold ``tau``.

Arrived in Phase 2 (CLAUDE.md §5). The gate uses the *margin*
``M_i = cos(s_hat_i, e_i) - max_{j!=i} cos(s_hat_i, e_j)`` -- never raw
similarity -- and is guarded *before* it by the ``V_i`` enrollment-variance
check, because a contaminated enrollment would happily pass its own gate
(CLAUDE.md §2). See :mod:`dagger.gate.confidence` for the combined decision,
:mod:`dagger.gate.enrollment` for ``V_i``, :mod:`dagger.gate.margin` for
``M_i``, and :mod:`dagger.gate.artifact` for the VAD/artifact checks.
"""

from __future__ import annotations

from dagger.gate.artifact import spectral_flatness, vad_coverage
from dagger.gate.confidence import GateResult, confidence_gate
from dagger.gate.enrollment import enrollment_variance_ok
from dagger.gate.margin import identity_margin

__all__ = [
    "GateResult",
    "confidence_gate",
    "enrollment_variance_ok",
    "identity_margin",
    "spectral_flatness",
    "vad_coverage",
]
