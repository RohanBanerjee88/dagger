"""Phase 1: differentiable SI-SDR loss and PIT (dagger.losses).

The silent-target masking behavior tested here is a direct regression guard
for the "degenerate loss terms drowned the training signal" bug (CLAUDE.md
Phase 1 KNOWN ISSUE, 2026-07-11): a ~silent crop target must not contribute a
``-10*log10(eps)`` garbage gradient to the mean.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dagger.losses.pit import pit_loss  # noqa: E402
from dagger.losses.sisdr import si_sdr_loss  # noqa: E402
from dagger.metrics.sisdr import si_sdr  # noqa: E402


class TestSiSdrLoss:
    def test_matches_metrics_si_sdr_numerically(self):
        rng = np.random.default_rng(0)
        target_np = rng.normal(size=16)
        estimate_np = target_np + 0.1 * rng.normal(size=16)
        expected = si_sdr(estimate_np, target_np)

        target = torch.tensor(target_np)
        estimate = torch.tensor(estimate_np)
        loss = si_sdr_loss(estimate, target)
        assert loss.item() == pytest.approx(-expected, rel=1e-4)

    def test_identical_signals_give_very_negative_loss(self):
        x = torch.tensor([1.0, -2.0, 3.0, 0.5])
        loss = si_sdr_loss(x, x)
        assert loss.item() < -50.0  # near -inf SI-SDR, clamped by eps

    def test_reduction_modes_shapes(self):
        estimate = torch.randn(4, 3, 16)
        target = torch.randn(4, 3, 16)
        none = si_sdr_loss(estimate, target, reduction="none")
        mean = si_sdr_loss(estimate, target, reduction="mean")
        summed = si_sdr_loss(estimate, target, reduction="sum")
        assert none.shape == (4, 3)
        assert mean.shape == ()
        assert summed.shape == ()
        assert summed.item() == pytest.approx(none.sum().item(), rel=1e-4)

    def test_gradient_flows_to_estimate(self):
        estimate = torch.randn(2, 8, requires_grad=True)
        target = torch.randn(2, 8)
        loss = si_sdr_loss(estimate, target)
        loss.backward()
        assert estimate.grad is not None
        assert torch.isfinite(estimate.grad).all()


class TestPitLoss:
    def test_picks_best_permutation_for_swapped_speakers(self):
        torch.manual_seed(0)
        targets = torch.randn(1, 2, 32)
        # estimates are targets, but in swapped order -> PIT must un-swap
        estimates = targets[:, [1, 0], :]
        loss = pit_loss(estimates, targets)
        assert loss.item() < -50.0  # near-perfect match under the best perm

    def test_worse_than_matching_the_wrong_permutation_directly(self):
        torch.manual_seed(1)
        targets = torch.randn(1, 2, 32)
        estimates = targets[:, [1, 0], :] + 0.5 * torch.randn(1, 2, 32)
        best = pit_loss(estimates, targets)
        wrong_order = si_sdr_loss(estimates, targets, reduction="mean")
        assert best.item() <= wrong_order.item()

    def test_silent_target_speaker_is_masked_out_of_the_mean(self):
        """A crop where speaker 2's target is ~silent must not contribute a
        degenerate -10*log10(eps) term to the loss."""
        torch.manual_seed(2)
        loud_target = torch.randn(1, 1, 32)
        silent_target = torch.zeros(1, 1, 32)
        targets = torch.cat([loud_target, silent_target], dim=1)  # [1, 2, 32]
        estimates = targets.clone()
        estimates[:, 0, :] = loud_target[:, 0, :]  # perfect estimate of the loud speaker
        estimates[:, 1, :] = torch.randn(1, 32)  # garbage on the silent speaker

        loss = pit_loss(estimates, targets)
        # If masking works, the loss is driven only by the perfectly-estimated
        # loud speaker (near -inf SI-SDR), not blown up by the silent one.
        assert loss.item() < -50.0

    def test_all_silent_batch_returns_zero_connected_to_graph(self):
        estimates = torch.randn(2, 2, 16, requires_grad=True)
        targets = torch.zeros(2, 2, 16)
        loss = pit_loss(estimates, targets)
        assert loss.item() == pytest.approx(0.0)
        loss.backward()  # must not raise
        assert estimates.grad is not None
