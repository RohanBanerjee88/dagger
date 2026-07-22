"""Confidence-gated embedding refinement + speaker ordering.

Arrived in Phase 2 (CLAUDE.md §5) as :mod:`dagger.refine.coarse_to_fine`.
Recursion refines ``e_bar_i`` (and, in general, processing order) ONLY -- it
must never feed a residual into ``G`` to produce output audio (CLAUDE.md §1).
See :func:`dagger.refine.coarse_to_fine.refine_embeddings`.
"""

from __future__ import annotations

from dagger.refine.coarse_to_fine import refine_embeddings

__all__ = ["refine_embeddings"]
