"""Phase 1: the TF-GridNet backbone, cross-attention fusion, the proposed
extractor G, and the blind-separation baseline (dagger.extract.*).

Uses tiny configs (small hidden size / FFT / one block) purely for speed --
these are architecture/wiring smoke tests, not quality tests (no training
happens here).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dagger.audio.provenance import ResidualInAudioPathError, original_mixture  # noqa: E402
from dagger.extract.blind import BlindSeparator, build_blind_separator_module  # noqa: E402
from dagger.extract.crossattn import build_speaker_conditioned_cross_attention  # noqa: E402
from dagger.extract.tfgridnet import build_backbone, match_length  # noqa: E402
from dagger.extract.tfgridnet_crossattn import (  # noqa: E402
    TFGridNetCrossAttnExtractor,
    build_tfgridnet_crossattn_module,
)

TINY_BACKBONE = dict(hidden_channels=8, n_blocks=1, n_fft=64, hop_length=16, n_heads=2)
LENGTH = 800


class TestMatchLength:
    def test_pads_short_waveform(self):
        out = match_length(np.ones(5), 8)
        assert out.shape == (8,)
        np.testing.assert_array_equal(out[5:], 0.0)

    def test_trims_long_waveform(self):
        out = match_length(np.ones(10), 6)
        assert out.shape == (6,)

    def test_noop_when_already_target_length(self):
        x = np.arange(5, dtype=np.float64)
        out = match_length(x, 5)
        np.testing.assert_array_equal(out, x)


class TestBuildBlock:
    def test_odd_hidden_channels_rejected(self):
        from dagger.extract.tfgridnet import build_block

        with pytest.raises(ValueError):
            build_block(hidden_channels=7)


class TestBuildBackbone:
    def test_forward_shapes(self):
        backbone = build_backbone(**TINY_BACKBONE)
        x = torch.randn(2, LENGTH)
        h, spec, length = backbone(x)
        assert length == LENGTH
        assert h.shape[0] == 2
        assert h.shape[1] == TINY_BACKBONE["hidden_channels"]
        assert torch.is_complex(spec)

    def test_istft_recovers_original_length(self):
        backbone = build_backbone(**TINY_BACKBONE)
        x = torch.randn(1, LENGTH)
        _, spec, length = backbone(x)
        out = backbone.istft(spec, length)
        assert out.shape[-1] == LENGTH


class TestSpeakerConditionedCrossAttention:
    def test_output_shape_matches_input(self):
        fusion = build_speaker_conditioned_cross_attention(
            embed_dim=16, feature_channels=8, n_tokens=2, n_heads=2
        )
        feats = torch.randn(2, 8, 5, 10)
        embedding = torch.randn(2, 16)
        out = fusion(feats, embedding)
        assert out.shape == feats.shape

    def test_different_embeddings_change_the_output(self):
        torch.manual_seed(0)
        fusion = build_speaker_conditioned_cross_attention(
            embed_dim=16, feature_channels=8, n_tokens=2, n_heads=2
        )
        feats = torch.randn(1, 8, 5, 10)
        emb_a = torch.randn(1, 16)
        emb_b = torch.randn(1, 16)
        out_a = fusion(feats, emb_a)
        out_b = fusion(feats, emb_b)
        assert not torch.allclose(out_a, out_b)


class TestTFGridNetCrossAttnModule:
    CFG = dict(
        **TINY_BACKBONE, embed_dim=16, n_tokens=2, cross_attn_blocks=1,
    )

    def test_forward_output_length(self):
        module = build_tfgridnet_crossattn_module(self.CFG)
        x = torch.randn(2, LENGTH)
        e = torch.randn(2, 16)
        out = module(x, e)
        assert out.shape[-1] == LENGTH or abs(out.shape[-1] - LENGTH) < TINY_BACKBONE["hop_length"]

    def test_gradient_flows_to_the_embedding_input(self):
        """Regression guard against the conditioning-collapse class of bug
        (CLAUDE.md Phase 1): at random init the fusion pathway must at least
        be wired so gradients reach the embedding, even before any training."""
        module = build_tfgridnet_crossattn_module(self.CFG)
        x = torch.randn(1, LENGTH)
        e = torch.randn(1, 16, requires_grad=True)
        out = module(x, e)
        out.sum().backward()
        assert e.grad is not None
        assert torch.any(e.grad != 0)

    def test_different_embeddings_perturb_the_output_at_init(self):
        torch.manual_seed(0)
        module = build_tfgridnet_crossattn_module(self.CFG)
        module.eval()
        x = torch.randn(1, LENGTH)
        e_a = torch.randn(1, 16)
        e_b = torch.randn(1, 16)
        with torch.no_grad():
            out_a = module(x, e_a)
            out_b = module(x, e_b)
        assert not torch.allclose(out_a, out_b)


class TestTFGridNetCrossAttnExtractor:
    CFG = dict(
        **TINY_BACKBONE, embed_dim=16, n_tokens=2, cross_attn_blocks=1,
    )

    def test_satisfies_extractor_interface_and_rejects_residual(self):
        extractor = TFGridNetCrossAttnExtractor(**self.CFG)
        x_O = original_mixture(np.random.randn(LENGTH))
        out = extractor.extract(x_O, embedding=np.random.randn(16))
        assert out.shape == (LENGTH,)

        residual = x_O - np.zeros(LENGTH)
        with pytest.raises(ResidualInAudioPathError):
            extractor.extract(residual, embedding=np.random.randn(16))

    def test_output_length_matches_input_exactly(self):
        extractor = TFGridNetCrossAttnExtractor(**self.CFG)
        x_O = original_mixture(np.random.randn(LENGTH + 7))  # not a multiple of hop_length
        out = extractor.extract(x_O, embedding=np.random.randn(16))
        assert out.shape == (LENGTH + 7,)


class TestBlindSeparator:
    CFG = dict(**TINY_BACKBONE, num_speakers=2)

    def test_module_forward_shape(self):
        module = build_blind_separator_module(self.CFG)
        x = torch.randn(2, LENGTH)
        out = module(x)
        assert out.shape[0] == 2
        assert out.shape[1] == 2  # num_speakers

    def test_separate_rejects_residual_input(self):
        separator = BlindSeparator(**self.CFG)
        x_O = original_mixture(np.random.randn(LENGTH))
        out = separator.separate(x_O, num_speakers=2)
        assert out.shape == (2, LENGTH)

        residual = x_O - np.zeros(LENGTH)
        with pytest.raises(ResidualInAudioPathError):
            separator.separate(residual, num_speakers=2)

    def test_separate_rejects_wrong_speaker_count(self):
        separator = BlindSeparator(**self.CFG)
        x_O = original_mixture(np.random.randn(LENGTH))
        with pytest.raises(ValueError):
            separator.separate(x_O, num_speakers=3)
