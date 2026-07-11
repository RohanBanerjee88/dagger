"""Permutation-invariant training loss, used only for the blind-separation baseline.

The target-conditioned extractor has a known target per speaker (the
embedding picks the speaker), so there is no permutation ambiguity for it.
:class:`~dagger.extract.blind.BlindSeparator` produces ``S`` outputs with no
such correspondence, so its training loss must search over permutations.
"""

from __future__ import annotations

import itertools
from typing import Callable


def pit_loss(estimates, targets, pairwise_fn: Callable = None):
    """Best-permutation loss over ``estimates``/``targets`` of shape ``[B, S, T]``.

    Brute-force over ``itertools.permutations(range(S))`` -- fine for the
    small ``S`` (2-3) this project uses; not a training-time bottleneck at
    that scale.
    """
    import torch

    from dagger.losses.sisdr import si_sdr_loss

    if pairwise_fn is None:
        pairwise_fn = si_sdr_loss

    b, s, _ = estimates.shape
    # A ~silent target (speaker absent from this crop) is unscoreable under a
    # scale-invariant loss -- it degenerates to -10*log10(eps) with a garbage
    # gradient. Mask those speakers out of the per-item mean; drop items whose
    # targets are all silent. The mask depends only on targets, so it is the
    # same for every permutation.
    valid = (targets * targets).sum(dim=-1) > 1e-8  # [B, S]
    item_valid = valid.any(dim=-1)  # [B]
    if not bool(item_valid.any()):
        # No scoreable item in the batch: a zero connected to the graph, so
        # callers can still call backward() (it just contributes no gradient).
        return estimates.sum() * 0.0
    weights = valid.to(estimates.dtype)  # [B, S]
    n_valid = weights.sum(dim=-1).clamp(min=1.0)  # [B]

    losses = []
    for perm in itertools.permutations(range(s)):
        permuted = estimates[:, list(perm), :]
        per_speaker = pairwise_fn(permuted, targets, reduction="none")  # [B, S]
        losses.append((per_speaker * weights).sum(dim=-1) / n_valid)  # [B]
    stacked = torch.stack(losses, dim=1)  # [B, n_perms]
    best, _ = stacked.min(dim=1)
    return best[item_valid].mean()
