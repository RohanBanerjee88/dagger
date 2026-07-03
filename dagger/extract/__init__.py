"""The extractor ``G(x_O, e_i)`` and its audio-path guard.

Phase 0 ships only :class:`NullExtractor` (no learning yet): it returns silence
on the overlap region, so reconstruction recovers solo regions exactly and
leaves overlaps empty. The real extractor (TF-GridNet + cross-attention fusion,
CLAUDE.md §3) arrives in Phase 1.

Every extractor runs its input through :func:`dagger.audio.provenance.
require_original_mixture`, so feeding a residual into ``G`` raises rather than
silently corrupting the result (guardrail §6.7).
"""

from dagger.extract.base import Extractor, NullExtractor

__all__ = ["Extractor", "NullExtractor"]
