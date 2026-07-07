"""Differentiable SI-SDR loss, mirroring ``dagger.metrics.sisdr.si_sdr``'s formula.

Used to train both the proposed extractor (per-speaker, known target -- see
:mod:`scripts.train_phase1`) and, wrapped in :mod:`dagger.losses.pit`, the
blind-separation baseline.
"""

from __future__ import annotations


def si_sdr_loss(estimate, target, eps: float = 1e-8, reduction: str = "mean"):
    """Negative SI-SDR (to minimize), batched over arbitrary leading dims.

    ``estimate``/``target``: ``[..., T]`` tensors. ``reduction``: ``"mean"``,
    ``"sum"``, or ``"none"`` (returns the un-reduced ``[...]`` tensor).
    """
    import torch

    target_energy = (target * target).sum(dim=-1, keepdim=True)
    scale = (estimate * target).sum(dim=-1, keepdim=True) / (target_energy + eps)
    projection = scale * target
    noise = estimate - projection

    ratio = (projection * projection).sum(dim=-1) / ((noise * noise).sum(dim=-1) + eps)
    neg_si_sdr = -10.0 * torch.log10(ratio + eps)

    if reduction == "none":
        return neg_si_sdr
    if reduction == "sum":
        return neg_si_sdr.sum()
    return neg_si_sdr.mean()
