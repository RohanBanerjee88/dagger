"""The blind-separation baseline: same backbone as G, no embedding conditioning.

Used only for comparison (CLAUDE.md Phase 1: "on 3+ speakers your method holds
up while the blind baseline visibly merges speakers"). Trained via
permutation-invariant loss (:mod:`dagger.losses.pit`) since it has no target
per output head. Deliberately *not* an :class:`~dagger.extract.base.Extractor`
subclass -- its signature is fundamentally different (no per-speaker
embedding input; produces all ``S`` outputs from one call).
"""

from __future__ import annotations

import numpy as np

from dagger.audio.provenance import TrackedSignal, require_original_mixture
from dagger.extract.tfgridnet import build_backbone, match_length

DEFAULT_CONFIG = {
    "hidden_channels": 32,
    "n_blocks": 4,
    "n_fft": 256,
    "hop_length": 64,
    "n_heads": 4,
    "num_speakers": 2,
}


def build_blind_separator_module(cfg: dict | None = None):
    """Build the trainable blind-separation ``nn.Module``.

    ``forward(x) -> [B, S, T]``, no embedding input. Shares
    :func:`~dagger.extract.tfgridnet.build_backbone` with the proposed
    extractor -- "same class of backbone" for a fair comparison -- but with
    ``num_speakers`` independent mask heads instead of one embedding
    -conditioned head.
    """
    import torch.nn as nn

    merged = {**DEFAULT_CONFIG, **(cfg or {})}
    num_speakers = merged["num_speakers"]

    class _BlindSeparatorModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_speakers = num_speakers
            self.backbone = build_backbone(
                hidden_channels=merged["hidden_channels"],
                n_blocks=merged["n_blocks"],
                n_fft=merged["n_fft"],
                hop_length=merged["hop_length"],
                n_heads=merged["n_heads"],
            )
            self.output_conv = nn.Conv2d(
                merged["hidden_channels"], 2 * num_speakers, kernel_size=1
            )

        def forward(self, x):
            import torch

            h, spec, length = self.backbone(x)  # no embedding, no fusion
            masks = self.output_conv(h)  # [B, 2*S, F, T]
            b = masks.shape[0]
            masks = masks.view(b, self.num_speakers, 2, masks.shape[2], masks.shape[3])
            outs = []
            for s in range(self.num_speakers):
                mask_complex = torch.complex(masks[:, s, 0], masks[:, s, 1])
                out_spec = spec * mask_complex
                outs.append(self.backbone.istft(out_spec, length))
            return torch.stack(outs, dim=1)  # [B, S, T]

    return _BlindSeparatorModule()


class BlindSeparator:
    """Inference-only wrapper around :func:`build_blind_separator_module`."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        **cfg,
    ) -> None:
        import torch

        self.device = device
        self.module = build_blind_separator_module(cfg)
        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=device)
            self.module.load_state_dict(state["state_dict"])
        self.module.to(device)
        self.module.eval()

    def separate(self, x: TrackedSignal, num_speakers: int) -> np.ndarray:
        """Separate all speakers from ``x`` at once; returns ``[S, T]``.

        ``x`` must be an original-mixture :class:`TrackedSignal` -- honoring
        the same no-residual discipline as :class:`~dagger.extract.base.Extractor`
        even though this baseline is diagnostic-only, so there is a single
        project-wide chokepoint for "mixture -> output audio".
        """
        samples = require_original_mixture(x, context="BlindSeparator.separate")
        if num_speakers != self.module.num_speakers:
            raise ValueError(
                f"BlindSeparator was built for {self.module.num_speakers} speakers, "
                f"got num_speakers={num_speakers}."
            )

        import torch

        with torch.no_grad():
            x_t = torch.as_tensor(samples, dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.module(x_t)
        out_np = out.squeeze(0).cpu().numpy().astype(np.float64)  # [S, T]
        return np.stack([match_length(row, samples.shape[0]) for row in out_np], axis=0)
