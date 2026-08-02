"""Tests for confidence-gate threshold selection (CLAUDE.md §2, Phase 2 "Change 2").

Two things are under test:

1. The :func:`gate_diagnostics` / :func:`apply_thresholds` split introduced so
   thresholds can be swept over recorded numbers instead of re-running the
   pipeline per candidate. The load-bearing property is that the split changed
   nothing: `apply_thresholds` must reproduce `confidence_gate`'s verdict AND
   its `reason` precedence exactly.
2. `scripts/tune_gate.py`'s sweep arithmetic, and the contaminated-enrollment
   fixture it depends on -- if contaminated enrollment does not actually raise
   `V_i`, that whole sweep measures nothing.

Everything here uses the suite's fake encoder and synthetic scenes; no corpus,
GPU, or checkpoint is needed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import FakeSpeakerEncoder, build_staggered_scene  # noqa: E402

from dagger.gate.confidence import (  # noqa: E402
    GateDiagnostics, apply_thresholds, confidence_gate, gate_diagnostics,
)
from dagger.gate.enrollment import enrollment_variance_ok, mean_enrollment_variance  # noqa: E402

SAMPLE_RATE = 8000

PASSING = dict(
    tau_margin=-1.0, max_mean_variance=1e9, min_vad_coverage=0.0, max_artifact_score=1e9,
)


def _load_tune_gate():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "tune_gate_under_test", repo_root / "scripts" / "tune_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tune_gate = _load_tune_gate()


class TestApplyThresholds:
    @pytest.mark.parametrize(
        "diagnostics,thresholds,expected_reason",
        [
            # V_i first: a bad enrollment rejects even when everything else passes.
            (GateDiagnostics(0.9, 5.0, 1.0, 0.0), {**PASSING, "max_mean_variance": 0.05},
             "enrollment_variance"),
            (GateDiagnostics(0.0, -5.0, 1.0, 0.0), {**PASSING, "tau_margin": 0.1}, "margin"),
            (GateDiagnostics(0.0, 5.0, 0.1, 0.0), {**PASSING, "min_vad_coverage": 0.5},
             "vad_coverage"),
            (GateDiagnostics(0.0, 5.0, 1.0, 0.99), {**PASSING, "max_artifact_score": 0.9},
             "artifact_score"),
            (GateDiagnostics(0.0, 5.0, 1.0, 0.0), PASSING, "accepted"),
        ],
    )
    def test_reason_names_the_first_failing_check(self, diagnostics, thresholds, expected_reason):
        result = apply_thresholds(diagnostics, **thresholds)

        assert result.reason == expected_reason
        assert result.accepted == (expected_reason == "accepted")

    def test_precedence_is_v_i_then_margin_then_coverage_then_artifact(self):
        """All four failing at once must report `enrollment_variance` -- the
        order encodes CLAUDE.md §2's "the gate can't check its own enrollment",
        and a caller reading `reason` relies on it."""
        everything_bad = GateDiagnostics(
            mean_variance=9.9, margin=-9.9, vad_coverage=0.0, artifact_score=9.9,
        )

        result = apply_thresholds(
            everything_bad, tau_margin=0.1, max_mean_variance=0.05,
            min_vad_coverage=0.5, max_artifact_score=0.9,
        )

        assert result.reason == "enrollment_variance"

    def test_nan_diagnostics_reject_rather_than_pass(self):
        """NaN compares False against anything, so an undefined diagnostic must
        fail its comparison instead of silently sailing through."""
        for field in ("mean_variance", "margin", "vad_coverage", "artifact_score"):
            values = {"mean_variance": 0.0, "margin": 5.0, "vad_coverage": 1.0,
                      "artifact_score": 0.0, field: float("nan")}
            result = apply_thresholds(
                GateDiagnostics(**values), tau_margin=0.1, max_mean_variance=0.05,
                min_vad_coverage=0.5, max_artifact_score=0.9,
            )
            assert result.accepted is False, f"NaN {field} was accepted"

    def test_records_the_variance_the_threshold_was_compared_against(self):
        """Without this value on the result there is no way to sweep
        `max_mean_variance` from a dumped CSV -- the threshold would be the one
        of the four with no measured number anywhere in a run."""
        result = apply_thresholds(GateDiagnostics(0.031, 5.0, 1.0, 0.0), **PASSING)

        assert result.mean_variance == pytest.approx(0.031)


class TestRefactorEquivalence:
    """`confidence_gate` must be unchanged by the split -- same verdict, same
    reason, for every combination of which check fails."""

    @staticmethod
    def _call(estimate, variance, thresholds, *, full_diagnostics):
        encoder = FakeSpeakerEncoder()
        return confidence_gate(
            estimate, SAMPLE_RATE, np.array([1.0, 0.0, 0.0]),
            [np.array([0.0, 1.0, 0.0])], encoder, variance,
            np.ones(estimate.shape[0], dtype=bool),
            full_diagnostics=full_diagnostics, **thresholds,
        )

    @pytest.mark.parametrize("max_mean_variance", [1e9, 1e-9])
    @pytest.mark.parametrize("tau_margin", [-1e9, 1e9])
    @pytest.mark.parametrize("min_vad_coverage", [0.0, 1.1])
    def test_verdict_matches_the_composed_path(
        self, max_mean_variance, tau_margin, min_vad_coverage
    ):
        estimate = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
        variance = np.full(3, 0.01)
        thresholds = dict(
            tau_margin=tau_margin, max_mean_variance=max_mean_variance,
            min_vad_coverage=min_vad_coverage, max_artifact_score=1e9,
        )

        short_circuited = self._call(estimate, variance, thresholds, full_diagnostics=False)
        computed = self._call(estimate, variance, thresholds, full_diagnostics=True)

        assert short_circuited.accepted == computed.accepted
        assert short_circuited.reason == computed.reason

    def test_full_diagnostics_fills_in_what_the_short_circuit_leaves_nan(self):
        """The reason a tuning run pays the extra encoder call: a V_i-rejected
        row is unusable for sweeping the other three thresholds if its margin
        and coverage were never computed."""
        estimate = np.sin(2 * np.pi * 220 * np.arange(SAMPLE_RATE) / SAMPLE_RATE)
        variance = np.full(3, 0.9)  # fails V_i
        thresholds = {**PASSING, "max_mean_variance": 0.05}

        short_circuited = self._call(estimate, variance, thresholds, full_diagnostics=False)
        computed = self._call(estimate, variance, thresholds, full_diagnostics=True)

        assert short_circuited.reason == computed.reason == "enrollment_variance"
        assert np.isnan(short_circuited.margin)
        assert not np.isnan(computed.margin)
        # The V_i value itself is recorded either way.
        assert short_circuited.mean_variance == pytest.approx(computed.mean_variance)


class TestEnrollmentVarianceHelpers:
    def test_ok_is_the_mean_compared_against_the_threshold(self):
        variance = np.array([0.01, 0.03, 0.02])

        assert mean_enrollment_variance(variance) == pytest.approx(0.02)
        assert enrollment_variance_ok(variance, 0.02) is True
        assert enrollment_variance_ok(variance, 0.019) is False


class TestContaminationFixture:
    def test_contaminated_mask_selects_overlap_not_solo(self):
        """The fixture the V_i sweep depends on: it must hand `enroll_speaker`
        audio from the speaker's OVERLAP region. If it accidentally returned
        solo audio, the sweep would compare a population against itself and
        report a meaningless 0% detection rate."""
        built = build_staggered_scene(
            lengths=[SAMPLE_RATE] * 2, overlap=0.5, min_solo=SAMPLE_RATE // 4,
            sample_rate=SAMPLE_RATE,
        )
        activity, solo, overlap = built["activity"], built["solo"], built["overlap"]

        mask = tune_gate._contaminated_mask(activity[0], overlap)

        assert mask.sum() > 0, "no overlap samples selected"
        assert np.all(mask[solo[0] > 0] == 0), "solo samples leaked into the contaminated mask"
        assert np.all((mask > 0) <= (overlap > 0)), "selected a sample outside the overlap region"


class TestSweeps:
    def _rows(self):
        # Healthy variance clusters low, contaminated clusters high, with a
        # deliberate overlap between 0.04 and 0.06 so no threshold is perfect.
        rows = []
        for value in (0.01, 0.02, 0.03, 0.06):
            rows.append({"population": tune_gate.HONEST, "mean_variance": value,
                         "margin": 1.0, "vad_coverage": 1.0, "artifact_score": 0.0})
        for value in (0.04, 0.10, 0.20, 0.50):
            rows.append({"population": tune_gate.CONTAMINATED, "mean_variance": value,
                         "margin": 1.0, "vad_coverage": 1.0, "artifact_score": 0.0})
        return rows

    def test_detection_and_false_rejection_rates(self):
        text = "\n".join(tune_gate._detection_sweep(
            self._rows(), "mean_variance", [0.05],
            healthy=tune_gate.HONEST, faulty=tune_gate.CONTAMINATED, reject_below=False,
        ))

        # At 0.05: contaminated above it = 0.10/0.20/0.50 -> 75%.
        #          honest above it       = 0.06            -> 25%.
        assert "| 0.05 | 75.0% | 25.0% | +0.500 |" in text

    def test_suggests_the_highest_youden_j(self):
        text = "\n".join(tune_gate._detection_sweep(
            self._rows(), "mean_variance", [0.005, 0.05, 0.3],
            healthy=tune_gate.HONEST, faulty=tune_gate.CONTAMINATED, reject_below=False,
        ))

        # 0.005 catches everything but rejects every honest row too (J=0);
        # 0.3 catches only 0.50 (J=0.25); 0.05 is the knee (J=0.50).
        assert "**suggested `mean_variance`: 0.05**" in text

    def test_reject_below_flips_the_comparison(self):
        """`tau_margin` rejects values BELOW it while `max_mean_variance`
        rejects values ABOVE it -- getting this backwards would silently invert
        every recommendation."""
        rows = [
            {"population": tune_gate.CORRECT, "margin": 1.0},
            {"population": tune_gate.SWAPPED, "margin": -1.0},
        ]

        text = "\n".join(tune_gate._detection_sweep(
            rows, "margin", [0.0],
            healthy=tune_gate.CORRECT, faulty=tune_gate.SWAPPED, reject_below=True,
        ))

        assert "| 0 | 100.0% | 0.0% | +1.000 |" in text

    def test_refuses_to_recommend_when_the_populations_do_not_separate(self):
        """A sweep over two identical distributions still has a "best" row --
        whichever came first in the grid. Recommending it would launder noise
        into a config value, so the report must refuse instead."""
        rows = [
            {"population": population, "mean_variance": value}
            for population in (tune_gate.HONEST, tune_gate.CONTAMINATED)
            for value in (0.01, 0.02, 0.03)
        ]

        text = "\n".join(tune_gate._detection_sweep(
            rows, "mean_variance", [0.005, 0.05, 0.5],
            healthy=tune_gate.HONEST, faulty=tune_gate.CONTAMINATED, reject_below=False,
        ))

        assert "NO USABLE THRESHOLD" in text
        assert "suggested" not in text

    def test_reports_population_medians_either_way(self):
        """The first thing to check when a sweep refuses is whether the fault
        fixture actually moved the distribution, so both medians are always
        printed."""
        text = "\n".join(tune_gate._detection_sweep(
            self._rows(), "mean_variance", [0.05],
            healthy=tune_gate.HONEST, faulty=tune_gate.CONTAMINATED, reject_below=False,
        ))

        assert "population medians -- honest: 0.02500, contaminated: 0.15000" in text

    def test_empty_population_does_not_crash_the_sweep(self):
        rows = [{"population": tune_gate.HONEST, "mean_variance": 0.01}]

        text = "\n".join(tune_gate._detection_sweep(
            rows, "mean_variance", [0.05],
            healthy=tune_gate.HONEST, faulty=tune_gate.CONTAMINATED, reject_below=False,
        ))

        assert "a population is empty" in text

    def test_rate_sweep_reports_healthy_rejection_only(self):
        rows = [
            {"population": tune_gate.CORRECT, "vad_coverage": c}
            for c in (0.2, 0.4, 0.6, 0.8)
        ]

        text = "\n".join(tune_gate._rate_sweep(
            rows, "vad_coverage", [0.5], reject_below=True,
        ))

        assert "| 0.5 | 50.0% |" in text
