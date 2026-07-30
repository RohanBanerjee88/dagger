"""Waveform-level energy normalization for the extractor G.

There is no waveform-level energy normalization anywhere else in the
pipeline: `dagger/data/mixing.py` sums independently-gained sources with no
renormalization, so the raw spectrogram magnitude `G` sees scales with
however many concurrent sources happen to be summed. Every training scene so
far summed at most `n_src` concurrent sources, so a deeper overlap is also,
mechanically, a louder/denser input than anything in the training
distribution -- on top of (not instead of) the genuine difficulty increase
from more simultaneous spectral masking. Wrapping `G`'s forward pass with
normalize-on-entry / denormalize-on-exit removes the amplitude-scale part of
that confound: the network's internal computation always runs at one
consistent scale, regardless of how many speakers are actually in the
mixture.

Computed only over the *active* (nonzero) region of a waveform, not the
whole tensor -- training crops are short and overlap-centered (mostly
nonzero) while inference feeds whole scenes masked to just the overlap
region (mostly zero outside it); a whole-tensor RMS would conflate "how much
of this tensor is zero padding" with "how many sources are summed," which is
exactly the confound this module exists to remove.
"""

from __future__ import annotations


def active_rms(x, eps: float = 1e-8):
    """Per-batch-item RMS of ``x`` over its nonzero samples only.

    ``x``: ``[B, T]``. Returns ``[B]``. A row with no nonzero samples at all
    (e.g. a scene with no overlap) returns a scale of exactly ``1.0`` rather
    than dividing by the ``eps`` floor, which would amplify floating-point
    noise in an all-silent input into something audible.
    """
    import torch

    active = x != 0
    count = active.sum(dim=-1).clamp_min(1)
    energy = (x * x).sum(dim=-1)
    rms = torch.sqrt(energy / count.to(x.dtype)).clamp_min(eps)
    all_zero = active.sum(dim=-1) == 0
    return torch.where(all_zero, torch.ones_like(rms), rms)


def normalize(x, eps: float = 1e-8):
    """``(x_norm, scale)`` with ``x_norm = x / scale``, ``scale = active_rms(x)``."""
    scale = active_rms(x, eps=eps)
    return x / scale.unsqueeze(-1), scale


def denormalize(x_norm, scale):
    """Invert :func:`normalize`: ``x_norm * scale``."""
    return x_norm * scale.unsqueeze(-1)
