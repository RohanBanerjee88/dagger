"""TF-GridNet backbone: an original implementation from the published
architecture description (Wang et al., "TF-GridNet: Making Time-Frequency
Domain Models Great Again for Monaural Speaker Separation"), not vendored or
copied from WeSep or USEF-TSE -- see CLAUDE.md Phase 1 notes on why (USEF-TSE
is CC-BY-NC 4.0, incompatible with this repo's Apache-2.0 license; WeSep ships
no license at all). This is a simplified, adapted reimplementation of the
block structure (intra-frequency modeling, inter-frame temporal modeling,
cross-frame attention), not a literal reproduction of any external codebase.

:class:`TFGridNetBackbone` is the single shared trunk used by both the
target-conditioned extractor (:mod:`dagger.extract.tfgridnet_crossattn`) and
the blind-separation baseline (:mod:`dagger.extract.blind`) -- "same class of
backbone" per CLAUDE.md's Phase 1 requirement for a fair proposed-vs-blind
comparison.
"""

from __future__ import annotations

import numpy as np


def match_length(waveform: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or trim ``waveform`` to exactly ``target_length`` samples.

    STFT/iSTFT framing can round to a length a few samples off from the input;
    every extractor output must match ``x_O``'s length exactly before it is
    stitched back into the reconstruction.
    """
    n = waveform.shape[-1]
    if n == target_length:
        return waveform
    if n > target_length:
        return waveform[..., :target_length]
    pad_width = [(0, 0)] * (waveform.ndim - 1) + [(0, target_length - n)]
    return np.pad(waveform, pad_width)


def _lazy_torch():
    import torch

    return torch


def build_block(hidden_channels: int, n_heads: int = 4):
    """One TF-GridNet block: intra-frequency + inter-frame + cross-frame attention.

    Returns an ``nn.Module`` instance; constructed lazily (imports ``torch``
    inside this function) so importing this file never requires ``torch`` to
    be installed.
    """
    torch = _lazy_torch()
    import torch.nn as nn

    class _TFGridNetBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            half = hidden_channels // 2
            self.intra_rnn = nn.LSTM(
                hidden_channels, half, batch_first=True, bidirectional=True
            )
            self.intra_norm = nn.GroupNorm(1, hidden_channels)
            self.inter_rnn = nn.LSTM(
                hidden_channels, half, batch_first=True, bidirectional=True
            )
            self.inter_norm = nn.GroupNorm(1, hidden_channels)
            self.attn = nn.MultiheadAttention(
                hidden_channels, n_heads, batch_first=True
            )
            self.attn_norm = nn.GroupNorm(1, hidden_channels)

        def forward(self, x):
            # x: [B, C, F, T]
            b, c, f, t = x.shape

            # Intra-frame full-band module: an RNN over frequency, independently
            # per time frame.
            intra_in = x.permute(0, 3, 2, 1).reshape(b * t, f, c)
            intra_out, _ = self.intra_rnn(intra_in)
            intra_out = intra_out.reshape(b, t, f, c).permute(0, 3, 2, 1)
            x = self.intra_norm(x + intra_out)

            # Sub-band temporal module: an RNN over time, independently per
            # frequency bin.
            inter_in = x.permute(0, 2, 3, 1).reshape(b * f, t, c)
            inter_out, _ = self.inter_rnn(inter_in)
            inter_out = inter_out.reshape(b, f, t, c).permute(0, 3, 1, 2)
            x = self.inter_norm(x + inter_out)

            # Cross-frame self-attention module: pool over frequency to get one
            # token per time frame, attend across time, broadcast back.
            tokens = x.mean(dim=2).transpose(1, 2)  # [B, T, C]
            attn_out, _ = self.attn(tokens, tokens, tokens)
            attn_out = attn_out.transpose(1, 2).unsqueeze(2)  # [B, C, 1, T]
            x = self.attn_norm(x + attn_out)
            return x

    return _TFGridNetBlock()


def build_backbone(
    hidden_channels: int = 32,
    n_blocks: int = 4,
    n_fft: int = 256,
    hop_length: int = 64,
    n_heads: int = 4,
):
    torch = _lazy_torch()
    import torch.nn as nn

    class _TFGridNetBackbone(nn.Module):
        """STFT -> input projection -> N :func:`build_block` blocks.

        ``forward`` optionally interleaves a caller-supplied ``fusion`` module
        (e.g. :class:`~dagger.extract.crossattn.SpeakerConditionedCrossAttention`)
        before the first ``cross_attn_blocks`` blocks, so the same backbone
        class serves both the target-conditioned extractor and the
        unconditioned blind-separation baseline.
        """

        def __init__(self) -> None:
            super().__init__()
            self.n_fft = n_fft
            self.hop_length = hop_length
            self.register_buffer("window", torch.hann_window(n_fft))
            self.input_conv = nn.Conv2d(2, hidden_channels, kernel_size=1)
            self.blocks = nn.ModuleList(
                [build_block(hidden_channels, n_heads) for _ in range(n_blocks)]
            )
            self.hidden_channels = hidden_channels

        def stft(self, x):
            return torch.stft(
                x, n_fft=self.n_fft, hop_length=self.hop_length,
                window=self.window, return_complex=True,
            )

        def istft(self, spec, length: int):
            return torch.istft(
                spec, n_fft=self.n_fft, hop_length=self.hop_length,
                window=self.window, length=length,
            )

        def forward(self, x, embedding=None, fusion=None, cross_attn_blocks: int = 0):
            length = x.shape[-1]
            spec = self.stft(x)  # [B, F, Tf] complex
            feat = torch.stack([spec.real, spec.imag], dim=1)  # [B, 2, F, Tf]
            h = self.input_conv(feat)
            for i, block in enumerate(self.blocks):
                if fusion is not None and embedding is not None and i < cross_attn_blocks:
                    h = fusion(h, embedding)
                h = block(h)
            return h, spec, length

    return _TFGridNetBackbone()
