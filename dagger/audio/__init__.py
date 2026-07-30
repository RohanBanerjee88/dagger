"""Audio primitives shared across modules.

Holds :mod:`dagger.audio.provenance`, the provenance-tracking wrapper that
enforces the audio-path rule from CLAUDE.md §1, and
:mod:`dagger.audio.normalize`, waveform-level energy normalization used by
the extractor `G`.
"""

from dagger.audio.normalize import active_rms, denormalize, normalize
from dagger.audio.provenance import (
    Provenance,
    ResidualInAudioPathError,
    TrackedSignal,
)

__all__ = [
    "Provenance",
    "ResidualInAudioPathError",
    "TrackedSignal",
    "active_rms",
    "denormalize",
    "normalize",
]
