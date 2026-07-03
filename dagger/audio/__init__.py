"""Audio primitives shared across modules.

Currently holds :mod:`dagger.audio.provenance`, the provenance-tracking
wrapper that enforces the audio-path rule from CLAUDE.md §1.
"""

from dagger.audio.provenance import (
    Provenance,
    ResidualInAudioPathError,
    TrackedSignal,
)

__all__ = ["Provenance", "ResidualInAudioPathError", "TrackedSignal"]
