"""Training losses: SI-SDR, speaker-consistency, noise-head reconstruction, artifact.

Not yet implemented. Arrives in Phase 1+. The reconstruction loss MUST keep a
noise term — ``||x_O - sum s_hat_i - n_hat||^2`` — or be trained noise-free, or
it fights the separation loss whenever noise != 0 (CLAUDE.md §2, guardrail §6.5).
See CLAUDE.md §5.
"""
