"""Tests for dagger.audio.normalize -- waveform-level energy normalization
used by the extractor G (see dagger/audio/normalize.py's docstring for why).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dagger.audio.normalize import active_rms, denormalize, normalize  # noqa: E402


class TestActiveRms:
    def test_ignores_zero_padding(self):
        active = torch.tensor([2.0, -2.0, 2.0, -2.0])  # rms = 2.0
        padded_short = torch.cat([active, torch.zeros(4)])
        padded_long = torch.cat([active, torch.zeros(100)])
        rms_short = active_rms(padded_short.unsqueeze(0))
        rms_long = active_rms(padded_long.unsqueeze(0))
        assert torch.allclose(rms_short, torch.tensor([2.0]))
        assert torch.allclose(rms_short, rms_long)

    def test_all_zero_row_returns_scale_one(self):
        x = torch.zeros(1, 16)
        rms = active_rms(x)
        assert torch.allclose(rms, torch.ones(1))

    def test_batched_rows_are_independent(self):
        x = torch.stack([
            torch.full((8,), 1.0),
            torch.full((8,), 3.0),
        ])
        rms = active_rms(x)
        assert torch.allclose(rms, torch.tensor([1.0, 3.0]))


class TestNormalizeDenormalize:
    def test_roundtrip_recovers_original(self):
        torch.manual_seed(0)
        x = torch.randn(3, 200)
        x_norm, scale = normalize(x)
        recovered = denormalize(x_norm, scale)
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_normalized_output_has_unit_active_rms(self):
        torch.manual_seed(0)
        x = torch.randn(2, 200) * 5.0
        x_norm, _ = normalize(x)
        assert torch.allclose(active_rms(x_norm), torch.ones(2), atol=1e-4)

    def test_scale_is_positively_homogeneous(self):
        """active_rms(k*x) == k*active_rms(x) for k > 0 -- the property that
        makes G's forward pass exactly scale-equivariant (see
        tests/phase1/test_extractors_torch.py's module-level test)."""
        torch.manual_seed(0)
        x = torch.randn(2, 200)
        for k in (0.1, 10.0, 100.0):
            assert torch.allclose(active_rms(k * x), k * active_rms(x), rtol=1e-4)
