"""Diarization → activity matrix and solo/overlap regions.

Phase 0 uses the *oracle* path only: read ground-truth RTTM instead of running
a diarizer (CLAUDE.md §5, Phase 0; guardrail §6.2 "oracle diarization first").
The real pyannote path is added in Phase 3.
"""

from dagger.diarize.oracle import (
    Segment,
    activity_matrix,
    overlap_mixture,
    read_rttm,
    solo_overlap_regions,
)

__all__ = [
    "Segment",
    "activity_matrix",
    "overlap_mixture",
    "read_rttm",
    "solo_overlap_regions",
]
