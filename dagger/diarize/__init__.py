"""Diarization → activity matrix and solo/overlap regions.

Phase 0-2 used the *oracle* path only: activity derived from the clean sources,
so ``a_i(t)`` is bit-exact (CLAUDE.md §5 Phase 0; guardrail §6.2 "oracle
diarization first, always"). Phase 3 adds the real path behind
:class:`~dagger.diarize.base.Diarizer`, so both run through one seam and the
mandatory oracle-beside-real comparison differs in nothing but the backend.

:class:`~dagger.diarize.pyannote_diarizer.PyannoteDiarizer` is deliberately NOT
re-exported here — importing it drags in the optional ``[diarize]`` extra, and
this package is imported by the numpy-only core path. Import it by module.
"""

from dagger.diarize.base import Diarizer
from dagger.diarize.mapping import ClusterMapping, map_clusters_to_speakers
from dagger.diarize.oracle import (
    OracleDiarizer,
    Segment,
    activity_matrix,
    overlap_depth,
    overlap_mixture,
    read_rttm,
    solo_overlap_regions,
)
from dagger.diarize.regions import Regions, scene_regions

__all__ = [
    "ClusterMapping",
    "Diarizer",
    "OracleDiarizer",
    "Regions",
    "Segment",
    "activity_matrix",
    "map_clusters_to_speakers",
    "overlap_depth",
    "overlap_mixture",
    "read_rttm",
    "scene_regions",
    "solo_overlap_regions",
]
