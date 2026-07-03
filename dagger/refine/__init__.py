"""Confidence-gated embedding refinement + speaker ordering.

Not yet implemented. Arrives in Phase 2. Recursion here refines ``e_bar_i`` and
the processing order ONLY — it must never feed a residual into ``G`` to produce
output audio (CLAUDE.md §1). See CLAUDE.md §5.
"""
