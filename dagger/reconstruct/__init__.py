"""Soft-mask stitching of solo + overlap into per-speaker outputs.

Implements the audio-path equation from CLAUDE.md §1:

    s_hat_i(t) = x(t)·w_Ei(t) + G(x_O(t), e_i)·w_Oi(t)

with smooth, crossfaded windows so ``w_Ei + w_Oi = 1`` across speaker ``i``'s
active region (partition of unity, §2 "soft masks at seams"). The extractor is
always fed ``x_O`` — the untouched overlap mixture — never a residual.
"""

from dagger.reconstruct.stitch import (
    crossfade_windows,
    reconstruct_all,
    reconstruct_speaker,
)

__all__ = ["crossfade_windows", "reconstruct_all", "reconstruct_speaker"]
