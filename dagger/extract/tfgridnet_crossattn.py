"""The proposed target-speaker extractor G: TF-GridNet + cross-attention.

:func:`build_tfgridnet_crossattn_module` is the single source of truth for the
architecture, used both for training (:mod:`scripts.train_phase1`, which
trains the ``nn.Module`` directly) and inference (:class:`TFGridNetCrossAttnExtractor`,
which wraps it to satisfy the :class:`~dagger.extract.base.Extractor`
interface).
"""

from __future__ import annotations

import numpy as np

from dagger.extract.base import Extractor
from dagger.extract.crossattn import build_speaker_conditioned_cross_attention
from dagger.extract.tfgridnet import build_backbone, match_length

DEFAULT_CONFIG = {
    "hidden_channels": 32,
    "n_blocks": 4,
    "n_fft": 256,
    "hop_length": 64,
    "n_heads": 4,
    "embed_dim": 192,  # TitaNet-Large's embedding dimension
    "n_tokens": 4,
    "cross_attn_blocks": 1,
}


def build_tfgridnet_crossattn_module(cfg: dict | None = None):
    """Build the trainable TF-GridNet + cross-attention ``nn.Module``.

    ``forward(x, embedding) -> waveform``, both ``[B, T]`` / ``[B, embed_dim]``.
    """
    import torch.nn as nn

    merged = {**DEFAULT_CONFIG, **(cfg or {})}

    class _TFGridNetCrossAttnModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.cross_attn_blocks = merged["cross_attn_blocks"]
            self.backbone = build_backbone(
                hidden_channels=merged["hidden_channels"],
                n_blocks=merged["n_blocks"],
                n_fft=merged["n_fft"],
                hop_length=merged["hop_length"],
                n_heads=merged["n_heads"],
            )
            self.fusion = build_speaker_conditioned_cross_attention(
                embed_dim=merged["embed_dim"],
                feature_channels=merged["hidden_channels"],
                n_tokens=merged["n_tokens"],
                n_heads=merged["n_heads"],
            )
            self.output_conv = nn.Conv2d(merged["hidden_channels"], 2, kernel_size=1)

        def forward(self, x, embedding):
            import torch

            h, spec, length = self.backbone(
                x, embedding=embedding, fusion=self.fusion,
                cross_attn_blocks=self.cross_attn_blocks,
            )
            mask = self.output_conv(h)
            mask_complex = torch.complex(mask[:, 0], mask[:, 1])
            out_spec = spec * mask_complex
            return self.backbone.istft(out_spec, length)

    return _TFGridNetCrossAttnModule()


class TFGridNetCrossAttnExtractor(Extractor):
    """Inference-only wrapper: the proposed G, satisfying the ``Extractor`` ABC.

    Because :meth:`Extractor.extract` already calls
    :func:`~dagger.audio.provenance.require_original_mixture` before
    :meth:`_extract` ever runs, this subclass inherits the no-residual guard
    automatically -- no changes to the ABC or to
    :func:`~dagger.reconstruct.stitch.reconstruct_all` were needed.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        **cfg,
    ) -> None:
        import torch

        self.device = device
        self.module = build_tfgridnet_crossattn_module(cfg)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=device)
            self.module.load_state_dict(state["state_dict"])
        self.module.to(device)
        self.module.eval()

    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        import torch

        with torch.no_grad():
            x_t = torch.as_tensor(x_O, dtype=torch.float32, device=self.device).unsqueeze(0)
            e_t = torch.as_tensor(embedding, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.module(x_t, e_t)
        out_np = out.squeeze(0).cpu().numpy().astype(np.float64)
        return match_length(out_np, x_O.shape[0])
