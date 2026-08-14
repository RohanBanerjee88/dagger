"""Shared evaluation machinery: one definition of "the four systems".

Lives in the package rather than in ``scripts/`` because more than one entrypoint
runs it — ``run_phase2.py`` (oracle diarization) and ``run_phase3.py`` (oracle vs.
real, several arms per scene). This project has already paid twice for the
alternative: three copies of the SI-SDR clipping rule meant the 2026-07-26
``+-inf`` bug had to be found and fixed separately in each, which is what
``dagger/metrics/phase2_scores.py`` exists to prevent. Same reasoning, one level up.
"""

from dagger.eval.systems import (
    DEFLATION_SYSTEMS,
    GATE_FIELDS,
    SCORE_FIELDS,
    SYSTEMS,
    accepted_before,
    deflation_order,
    make_gate_fn,
    score_scene,
)

__all__ = [
    "DEFLATION_SYSTEMS",
    "GATE_FIELDS",
    "SCORE_FIELDS",
    "SYSTEMS",
    "accepted_before",
    "deflation_order",
    "make_gate_fn",
    "score_scene",
]
