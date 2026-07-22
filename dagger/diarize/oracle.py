"""Oracle diarization: ground-truth RTTM -> activity matrix -> regions.

Phase 0 never runs a diarizer. It reads the ground-truth RTTM so downstream
plumbing (enrollment regions, overlap mixture, reconstruction, metrics) can be
validated with a perfect diarization signal. Only once every module is correct
under oracle diarization do we swap in a real diarizer (Phase 3) — that is the
only way to attribute a failure to the diarizer vs. ``phi`` vs. ``G``
(CLAUDE.md guardrail §6.2).

Frame/sample alignment matters: the activity matrix is sampled at the *audio*
sample rate so masks line up with waveform samples exactly. A framerate/sample
-rate mismatch here is the Phase 0 "red flag" from CLAUDE.md §5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dagger.audio.provenance import TrackedSignal, original_mixture


@dataclass(frozen=True)
class Segment:
    """A single speaker-active span from an RTTM file."""

    speaker: str
    start: float  # seconds
    duration: float  # seconds

    @property
    def end(self) -> float:
        return self.start + self.duration


def read_rttm(path: str) -> list[Segment]:
    """Parse an NIST RTTM file into a list of :class:`Segment`.

    Only ``SPEAKER`` lines are used. Columns (space-separated):
    ``type file chan tbeg tdur ortho stype name conf slat``. We read
    ``tbeg`` (col 3), ``tdur`` (col 4) and the speaker ``name`` (col 7).
    """
    segments: list[Segment] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(";;") or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            segments.append(Segment(speaker=speaker, start=start, duration=duration))
    return segments


def speaker_order(segments: list[Segment]) -> list[str]:
    """Stable, deterministic speaker ordering (first appearance in the RTTM)."""
    seen: list[str] = []
    for seg in segments:
        if seg.speaker not in seen:
            seen.append(seg.speaker)
    return seen


def activity_matrix(
    segments: list[Segment],
    num_samples: int,
    sample_rate: int,
    speakers: list[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build the binary activity matrix ``a_i(t)`` of shape ``[S, num_samples]``.

    Returns ``(activity, speakers)`` where ``speakers[i]`` labels row ``i``.
    Segment boundaries are rounded to the nearest sample and clipped to
    ``[0, num_samples]`` so the mask aligns with the waveform exactly.
    """
    if speakers is None:
        speakers = speaker_order(segments)
    index = {spk: i for i, spk in enumerate(speakers)}
    activity = np.zeros((len(speakers), num_samples), dtype=np.float64)
    for seg in segments:
        if seg.speaker not in index:
            continue
        lo = int(round(seg.start * sample_rate))
        hi = int(round(seg.end * sample_rate))
        lo = max(0, min(lo, num_samples))
        hi = max(0, min(hi, num_samples))
        activity[index[seg.speaker], lo:hi] = 1.0
    return activity, speakers


def overlap_depth(activity: np.ndarray) -> np.ndarray:
    """Per-sample count of concurrently active speakers ``|K|(t)``, shape ``[T]``.

    This is the quantity Phase 2's depth-stratified metrics need (CLAUDE.md §5
    Phase 2: "stratify every metric by overlap depth |K|"). ``solo_overlap_regions``
    computes the same sum internally but only keeps its binarization
    (``overlap = depth >= 2``); this function exposes the raw count.
    """
    return activity.sum(axis=0).astype(np.int64)


def solo_overlap_regions(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split activity into per-speaker solo masks and a shared overlap mask.

    * ``solo[i, t] == 1``  iff speaker ``i`` is the *only* active speaker at ``t``
      (region ``E_i`` — clean, valid for enrollment and copied straight through).
    * ``overlap[t] == 1``  iff two or more speakers are active at ``t``
      (region where the extractor ``G`` must operate).

    Returns ``(solo, overlap)`` with shapes ``[S, T]`` and ``[T]``.
    """
    depth = activity.sum(axis=0)  # number of concurrent speakers per sample
    overlap = (depth >= 2).astype(np.float64)
    solo_frames = (depth == 1).astype(np.float64)
    solo = activity * solo_frames[None, :]
    return solo, overlap


def overlap_mixture(
    x: TrackedSignal | np.ndarray,
    overlap: np.ndarray,
    label: str = "x_O",
) -> TrackedSignal:
    """Build ``x_O`` = the mixture restricted to overlap frames.

    Crucially this returns an *original-mixture* :class:`TrackedSignal`: a masked
    slice of the mixture is still the mixture, so it is a legal extractor input.
    """
    if not isinstance(x, TrackedSignal):
        x = original_mixture(np.asarray(x))
    return x.masked(overlap, label=label)
