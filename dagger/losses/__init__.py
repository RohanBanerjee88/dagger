"""Training losses: SI-SDR, PIT (blind baseline), speaker-consistency, noise-head, artifact.

Phase 1 implements ``dagger.losses.sisdr`` (differentiable SI-SDR, used to
train the proposed extractor directly and, via ``dagger.losses.pit``, the
blind-separation baseline). Speaker-consistency and artifact losses remain
unimplemented.

The noise-head reconstruction loss (``||x_O - sum s_hat_i - n_hat||^2``) is
deliberately deferred, not yet implemented: WSJ0-2mix/LibriMix scenes
(``dagger.data.mixing.mix_sources``) are anechoic sums with no noise term by
construction, so Phase 1 trains noise-free (CLAUDE.md §2's explicit "or train
on noise-free data" branch). This noise head MUST land before Phase 3 trains
on real/noisy corpora, or the reconstruction loss will fight the separation
loss whenever noise != 0 (guardrail §6.5).
"""
