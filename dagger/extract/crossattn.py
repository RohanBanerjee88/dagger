"""Speaker-conditioned cross-attention fusion.

This repo's extractor interface is committed to a single mean embedding
``e_bar_i`` per speaker (``G(x_O, e_bar_i)``, CLAUDE.md's own equation), not a
frame-aligned reference encoding. The real USEF-TSE mechanism (arXiv:2409.02615)
conditions on a frame-level reference via cross-attention replacing
self-attention in the first M blocks -- not reproducible here as-is given the
fixed-embedding interface, and its code/weights are CC-BY-NC 4.0 regardless
(see CLAUDE.md Phase 1 notes). This module is a deliberate, necessary
adaptation: project the single embedding into a handful of learned "speaker
tokens", cross-attend the mixture features against them once, and FiLM
-modulate -- informed by the paper's architecture, not copied from it.
"""

from __future__ import annotations


def build_speaker_conditioned_cross_attention(
    embed_dim: int,
    feature_channels: int,
    n_tokens: int = 4,
    n_heads: int = 4,
):
    import torch.nn as nn

    class _SpeakerConditionedCrossAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_tokens = n_tokens
            self.feature_channels = feature_channels
            self.token_proj = nn.Linear(embed_dim, feature_channels * n_tokens)
            self.attn = nn.MultiheadAttention(feature_channels, n_heads, batch_first=True)
            self.film = nn.Linear(feature_channels, feature_channels * 2)

        def forward(self, mixture_feats, embedding):
            # mixture_feats: [B, C, F, T], embedding: [B, embed_dim]
            b, c, f, t = mixture_feats.shape
            # TitaNet embeddings arrive unnormalized; without this, their
            # magnitude spread makes early FiLM scales noisy and the optimizer
            # learns to mute the conditioning pathway (CLAUDE.md Phase 1
            # conditioning-collapse issue).
            embedding = nn.functional.normalize(embedding, dim=-1)
            tokens = self.token_proj(embedding).view(b, self.n_tokens, c)
            query = mixture_feats.permute(0, 2, 3, 1).reshape(b, f * t, c)
            attn_out, _ = self.attn(query, tokens, tokens)
            scale, shift = self.film(attn_out).chunk(2, dim=-1)
            modulated = query * (1.0 + scale) + shift
            modulated = modulated.view(b, f, t, c).permute(0, 3, 1, 2)
            return mixture_feats + modulated

    return _SpeakerConditionedCrossAttention()
