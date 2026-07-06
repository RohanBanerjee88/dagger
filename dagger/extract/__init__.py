"""The extractor ``G(x_O, e_i)`` and its audio-path guard.

Phase 0 shipped only :class:`NullExtractor` (no learning yet). Phase 1 adds the
real extractor, :class:`~dagger.extract.tfgridnet_crossattn.TFGridNetCrossAttnExtractor`
(an original TF-GridNet + cross-attention implementation, CLAUDE.md §3), and a
blind-separation baseline, :class:`~dagger.extract.blind.BlindSeparator`, for
comparison. Torch is imported lazily inside these modules' functions, so
``import dagger.extract`` never requires the ``[ml]`` extra.

Every :class:`Extractor` runs its input through :func:`dagger.audio.provenance.
require_original_mixture`, so feeding a residual into ``G`` raises rather than
silently corrupting the result (guardrail §6.7). :class:`~dagger.extract.blind.
BlindSeparator` honors the same guard even though it is diagnostic-only.
"""

from dagger.extract.base import Extractor, NullExtractor

__all__ = ["Extractor", "NullExtractor"]
