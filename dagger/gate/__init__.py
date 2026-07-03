"""Confidence gate: identity margin ``M_i``, VAD, artifact score, threshold ``tau``.

Not yet implemented. Arrives in Phase 2. The gate uses the *margin*
``M_i = cos(s_hat_i, e_i) - max_{j!=i} cos(s_hat_i, e_j)`` — never raw
similarity — and is guarded *before* it by the ``V_i`` enrollment-variance
check, because a contaminated enrollment would happily pass its own gate
(CLAUDE.md §2). See CLAUDE.md §5.
"""
