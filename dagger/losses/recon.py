"""L_recon -- the reconstruction term. The one term both documents write out.

    S10:          L_recon = || x_O - sum_{i in K} s_hat_i ||^2
    Corollary 1:  L_recon = || x_O - sum_i s_hat_i - n_hat ||^2

The noise head n_hat is omitted here, per Corollary 1's second branch:
LibriMix/WSJ0 scenes (dagger.data.mixing.mix_sources) are anechoic sums with
n == 0, so the un-headed term already has the truth as a common minimiser with
the separation loss. n_hat becomes mandatory for real/noisy corpora (Phase 4).

Note precisely what that branch licenses, because this package got it wrong
once: Corollary 1 says "on synthetic noise-free training data (n == 0) the
original loss is already consistent", where "the original loss" is L_recon
WITHOUT n_hat. It licenses omitting the noise head, never the term. Dropping
the whole term is what left output level unconstrained.

Unlike every other quantity in this project, this term is NOT scale-invariant,
and that is the point: it is the only term in the four-term objective that can
constrain output level. L_sep (SI-SDR) fits the scale out before measuring;
cosine similarity between unit-norm embeddings cannot see level; spectral
flatness scales its numerator and denominator alike.

USE WITH THE PER-SPEAKER BACKWARD. scripts/train_phase1.py backwards once per
speaker to avoid holding num_speakers TF-GridNet graphs. Build the stack with
exactly one speaker attached and the rest detached: detach changes no values,
so the residual is identical for every i, and gradients accumulate across
backward() calls to give the exact gradient of the true L_recon. Cost is 2x
forward compute and zero extra memory.
"""

from __future__ import annotations


def recon_loss(
    overlap_mixture,
    estimates,
    active=None,
    noise_estimate=None,
    normalize: bool = True,
    eps: float = 1e-8,
    reduction: str = "mean",
):
    """Squared reconstruction residual against the overlap mixture.

    ``overlap_mixture``: ``[B, T]``. ``estimates``: ``[B, S, T]``.

    ``active``: optional ``[B, S]`` bool mask selecting the active set K.
    Speakers outside K contribute nothing to the sum. PASS THIS: without it the
    sum runs over all S, and a speaker absent from the crop has a garbage
    estimate that lands straight in the residual -- the same silent-target trap
    ``dagger.losses.pit`` guards with its own mask.

    ``noise_estimate``: optional ``[B, T]``; leave ``None`` on noise-free
    corpora, which is every corpus this project trains on today.

    ``normalize``: divide by ``||x_O||^2``, making the term dimensionless so
    ``lambda_recon`` is comparable against the dB-scaled separation loss and
    need not be retuned per corpus. Both properties that matter survive: still
    exactly 0 at the truth, and a uniform gain ``c`` still costs ``(1 - c)^2``.
    This is a deviation from the documents' literal formula -- state it in the
    config rather than inheriting it (CLAUDE.md S7).

    ``reduction``: ``"mean"``, ``"sum"``, or ``"none"`` (returns ``[B]``).
    """
    import torch

    if active is not None:
        estimates = estimates * active.unsqueeze(-1).to(estimates.dtype)

    residual = overlap_mixture - estimates.sum(dim=1)
    if noise_estimate is not None:
        residual = residual - noise_estimate

    per_item = (residual * residual).sum(dim=-1)  # [B]
    if normalize:
        per_item = per_item / ((overlap_mixture * overlap_mixture).sum(dim=-1) + eps)

    if reduction == "none":
        return per_item
    if reduction == "sum":
        return per_item.sum()
    return torch.mean(per_item)
