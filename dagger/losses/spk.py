"""L_spk -- speaker consistency. NOT IMPLEMENTED: no formula exists.

A bare symbol in both documents. The gloss "push phi(s_hat_i) toward e_bar_i"
that appears in journal/phase3.md is OUR reading of the name, not a quotation;
it is sourced nowhere. L_spk is in exactly the same state as L_art, not a
better one.

TWO TESTS ANY CANDIDATE MUST PASS.

1. VANISH AT THE TRUTH (Proposition 7 / Corollary 1), or it fights L_sep.
   This kills the obvious form: 1 - cos(phi(s_hat_i), e_bar_i) is NOT zero at
   s_hat_i == s_i, because e_bar_i is a mean over Top-K solo clips (Eq. 18),
   not phi(s_i), so cos(phi(s_i), e_bar_i) < 1. It would carry a live gradient
   at the truth, pulling the estimate toward whatever signal embeds exactly at
   e_bar_i -- which is not the speaker. Corollary 1's repair reapplied gives a
   deviation form that does vanish:

       ( cos(phi(s_hat_i), e_bar_i) - cos(phi(s_i), e_bar_i) )^2

   with the second term constant in theta, computed once under no_grad. It also
   keeps e_bar_i in the expression, so it still targets the quantity the
   tau_margin gate reads.

2. ORDER DAMAGED ESTIMATES MONOTONICALLY against SI-SDR. Vanishing at the truth
   says nothing about the gradient anywhere else; this is exactly how the L_art
   candidate failed (see dagger/losses/art.py).

OPEN DESIGN QUESTION: the margin is cos(s_hat_i, e_i) - max_{j!=i}
cos(s_hat_i, e_j), a CONTRASTIVE quantity. A purely attractive term never
touches the max_{j!=i} half, so the predicted 0.019 -> 0.243 margin migration
may only partly move even if the loss works. Whether L_spk needs a repulsive
term is undecided.

IMPLEMENTATION BLOCKER, worse than journal/phase3.md records. phi has no
gradient path: TitaNetEncoder.embed (dagger/enroll/encoder.py:140-150) stacks
THREE separate breaks -- a numpy resample, a round-trip through a temp WAV file
on disk, and an explicit torch.no_grad(). A differentiable forward path
(torchaudio.functional.resample + the NeMo forward, weights frozen, gradients
to the input only) has to be built first. That is a new component, not a
one-line swap.
"""

from __future__ import annotations


def spk_loss(*args, **kwargs):
    raise NotImplementedError(
        "L_spk has no formula in either theory document. See this module's "
        "docstring for the two acceptance tests any candidate must pass, and "
        "for the three stacked breaks in phi's gradient path."
    )
