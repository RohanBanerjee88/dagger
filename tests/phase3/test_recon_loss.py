"""Stage C: ``L_recon`` must charge the level error SI-SDR structurally cannot see.

``G`` emitted the overlap region at a median 2.86x the true amplitude for three
phases (level_error_db +9.14, 2026-08-23). Root cause: ``si_sdr_loss`` is
scale-invariant and was the only term in the objective, so nothing constrained
output level. ``L_recon`` is the only term in the theory's four-term objective
that is not scale-invariant.

The load-bearing test here is
``test_charges_the_2_86x_error_that_si_sdr_scores_as_perfect``: it asserts the
DISAGREEMENT between the two losses on one input, which is the only way this
project has ever caught a scale defect (CLAUDE.md S7, S9).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dagger.losses.recon import recon_loss  # noqa: E402
from dagger.losses.sisdr import si_sdr_loss  # noqa: E402


def _scene(seed: int = 0, b: int = 2, s: int = 3, t: int = 4000):
    """A noise-free anechoic scene: ``x_O`` is exactly the sum of its sources."""
    generator = torch.Generator().manual_seed(seed)
    sources = torch.randn(b, s, t, generator=generator)
    return sources, sources.sum(dim=1)


class TestReconSeesLevel:
    def test_zero_at_the_truth(self):
        sources, overlap = _scene()
        assert float(recon_loss(overlap, sources)) < 1e-6

    def test_charges_the_2_86x_error_that_si_sdr_scores_as_perfect(self):
        sources, overlap = _scene()
        loud = 2.86 * sources

        # SI-SDR fits the scale out before measuring, so it sees no error.
        per_speaker = si_sdr_loss(loud, sources, reduction="none")
        assert float(per_speaker.max()) < -50.0  # i.e. SI-SDR > 50 dB

        # L_recon charges (1 - c)^2 of the mixture energy for a uniform gain c.
        assert float(recon_loss(overlap, loud)) == pytest.approx(1.86**2, rel=0.01)

    def test_penalty_is_monotone_in_the_level_error(self):
        sources, overlap = _scene()
        scores = [
            float(recon_loss(overlap, c * sources)) for c in (1.0, 1.2, 1.5, 2.0, 2.86)
        ]
        assert scores == sorted(scores)

    def test_normalize_off_keeps_the_documents_literal_formula(self):
        sources, overlap = _scene()
        raw = float(recon_loss(overlap, 2.86 * sources, normalize=False))
        energy = float((overlap * overlap).sum(dim=-1).mean())
        assert raw == pytest.approx(1.86**2 * energy, rel=0.01)


class TestActiveSet:
    def test_active_mask_restricts_the_sum_to_K(self):
        sources, _ = _scene()
        active = torch.ones(sources.shape[:2], dtype=torch.bool)
        active[:, -1] = False  # speaker absent from this crop
        overlap = sources[:, :-1].sum(dim=1)

        assert float(recon_loss(overlap, sources, active=active)) < 1e-6

    def test_without_the_mask_an_absent_speaker_lands_in_the_residual(self):
        sources, _ = _scene()
        overlap = sources[:, :-1].sum(dim=1)

        assert float(recon_loss(overlap, sources)) > 0.3


class TestNoiseHead:
    def test_noise_estimate_is_subtracted_when_supplied(self):
        sources, overlap = _scene()
        noise = 0.1 * torch.randn_like(overlap)

        # Phase 4 shape: the truth (sources, noise) zeroes the term.
        assert float(recon_loss(overlap + noise, sources, noise_estimate=noise)) < 1e-6
