"""Tests for the Phase 2 confidence gate (CLAUDE.md §2, §5)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import FakeSpeakerEncoder, make_tone  # noqa: E402

import dagger.gate.confidence as confidence_mod
from dagger.gate.artifact import spectral_flatness, vad_coverage
from dagger.gate.confidence import GateResult, confidence_gate
from dagger.gate.enrollment import enrollment_variance_ok
from dagger.gate.margin import identity_margin

SAMPLE_RATE = 8000


class TestEnrollmentVarianceGate:
    def test_low_variance_passes(self):
        variance = np.array([0.001, 0.002, 0.0005])
        assert enrollment_variance_ok(variance, max_mean_variance=0.01) is True

    def test_high_variance_fails(self):
        variance = np.array([1.0, 2.0, 0.5])
        assert enrollment_variance_ok(variance, max_mean_variance=0.01) is False

    def test_boundary_is_inclusive(self):
        variance = np.array([0.01, 0.01])
        assert enrollment_variance_ok(variance, max_mean_variance=0.01) is True


class TestIdentityMargin:
    def test_matching_speaker_has_positive_margin(self, fake_encoder):
        tone_a = make_tone(4000, 220.0)
        tone_b = make_tone(4000, 880.0)
        e_a = fake_encoder.embed(tone_a, SAMPLE_RATE)
        e_b = fake_encoder.embed(tone_b, SAMPLE_RATE)
        margin = identity_margin(tone_a, SAMPLE_RATE, e_a, [e_b], fake_encoder)
        assert margin > 0.0

    def test_mismatched_speaker_has_lower_margin(self, fake_encoder):
        tone_a = make_tone(4000, 220.0)
        tone_b = make_tone(4000, 880.0)
        e_a = fake_encoder.embed(tone_a, SAMPLE_RATE)
        e_b = fake_encoder.embed(tone_b, SAMPLE_RATE)
        margin_correct = identity_margin(tone_a, SAMPLE_RATE, e_a, [e_b], fake_encoder)
        margin_wrong = identity_margin(tone_b, SAMPLE_RATE, e_a, [e_b], fake_encoder)
        assert margin_correct > margin_wrong

    def test_no_other_speakers_is_nan(self, fake_encoder):
        tone_a = make_tone(4000, 220.0)
        e_a = fake_encoder.embed(tone_a, SAMPLE_RATE)
        margin = identity_margin(tone_a, SAMPLE_RATE, e_a, [], fake_encoder)
        assert np.isnan(margin)


class TestArtifactChecks:
    def test_vad_coverage_full_when_estimate_matches_expected_active(self):
        tone = make_tone(4000, 220.0, amp=0.8)
        expected_active = np.ones(4000, dtype=bool)
        coverage = vad_coverage(tone, expected_active, SAMPLE_RATE)
        assert coverage > 0.9

    def test_vad_coverage_low_for_silent_estimate(self):
        silence = np.zeros(4000)
        expected_active = np.ones(4000, dtype=bool)
        coverage = vad_coverage(silence, expected_active, SAMPLE_RATE)
        assert coverage == 0.0

    def test_vad_coverage_nan_when_nothing_expected_active(self):
        tone = make_tone(4000, 220.0)
        expected_active = np.zeros(4000, dtype=bool)
        assert np.isnan(vad_coverage(tone, expected_active, SAMPLE_RATE))

    def test_vad_coverage_handles_clips_shorter_than_the_default_analysis_window(self):
        # A clip shorter than active_mask's default win_ms=25ms (200 samples at
        # 8kHz) must not crash -- this is the shape a short refinement-round
        # overlap run can have (see dagger.refine.coarse_to_fine).
        short = np.array([3.0, 4.0])
        expected_active = np.ones(2, dtype=bool)
        coverage = vad_coverage(short, expected_active, SAMPLE_RATE)
        assert not np.isnan(coverage)
        assert 0.0 <= coverage <= 1.0

    def test_spectral_flatness_lower_for_tone_than_noise(self):
        rng = np.random.default_rng(0)
        tone = make_tone(4000, 220.0, amp=0.8)
        noise = rng.normal(size=4000)
        assert spectral_flatness(tone) < spectral_flatness(noise)

    def test_spectral_flatness_nan_for_short_signal(self):
        assert np.isnan(spectral_flatness(np.zeros(10), n_fft=512))


class TestConfidenceGate:
    _kwargs = dict(tau_margin=0.0, max_mean_variance=0.01, min_vad_coverage=0.5, max_artifact_score=0.9)

    def test_good_estimate_is_accepted(self, fake_encoder):
        tone_a = make_tone(4000, 220.0, amp=0.8)
        tone_b = make_tone(4000, 880.0, amp=0.8)
        e_a = fake_encoder.embed(tone_a, SAMPLE_RATE)
        e_b = fake_encoder.embed(tone_b, SAMPLE_RATE)
        result = confidence_gate(
            tone_a, SAMPLE_RATE, e_a, [e_b], fake_encoder,
            enrollment_variance=np.array([0.001]), expected_active=np.ones(4000, dtype=bool),
            **self._kwargs,
        )
        assert isinstance(result, GateResult)
        assert result.accepted is True
        assert result.reason == "accepted"

    def test_bad_enrollment_variance_short_circuits_before_margin(self, fake_encoder):
        tone_a = make_tone(4000, 220.0)
        e_a = fake_encoder.embed(tone_a, SAMPLE_RATE)
        result = confidence_gate(
            tone_a, SAMPLE_RATE, e_a, [e_a], fake_encoder,
            enrollment_variance=np.array([10.0]), expected_active=np.ones(4000, dtype=bool),
            **self._kwargs,
        )
        assert result.accepted is False
        assert result.reason == "enrollment_variance"
        assert np.isnan(result.margin)  # never computed

    def test_wrong_speaker_estimate_fails_on_margin(self, fake_encoder):
        tone_a = make_tone(4000, 220.0, amp=0.8)
        tone_b = make_tone(4000, 880.0, amp=0.8)
        e_a = fake_encoder.embed(tone_a, SAMPLE_RATE)
        e_b = fake_encoder.embed(tone_b, SAMPLE_RATE)
        result = confidence_gate(
            tone_b, SAMPLE_RATE, e_a, [e_b], fake_encoder,
            enrollment_variance=np.array([0.001]), expected_active=np.ones(4000, dtype=bool),
            **{**self._kwargs, "tau_margin": 0.5},
        )
        assert result.accepted is False
        assert result.reason == "margin"

    def test_low_coverage_fails_on_vad_coverage_not_margin(self, monkeypatch, fake_encoder):
        # Isolates the vad_coverage branch from the specific numeric behavior of
        # identity_margin/spectral_flatness (already covered above/in isolation):
        # a passing margin with a failing coverage must reject with reason
        # "vad_coverage", never masked by an earlier check.
        monkeypatch.setattr(confidence_mod, "identity_margin", lambda *a, **k: 1.0)
        monkeypatch.setattr(confidence_mod, "vad_coverage", lambda *a, **k: 0.0)
        monkeypatch.setattr(confidence_mod, "spectral_flatness", lambda *a, **k: 0.0)
        result = confidence_gate(
            make_tone(4000, 220.0), SAMPLE_RATE, np.zeros(3), [np.zeros(3)], fake_encoder,
            enrollment_variance=np.array([0.001]), expected_active=np.ones(4000, dtype=bool),
            **self._kwargs,
        )
        assert result.accepted is False
        assert result.reason == "vad_coverage"
        assert result.margin == 1.0  # already computed by the time coverage rejects

    def test_high_artifact_score_fails_on_artifact_score(self, monkeypatch, fake_encoder):
        monkeypatch.setattr(confidence_mod, "identity_margin", lambda *a, **k: 1.0)
        monkeypatch.setattr(confidence_mod, "vad_coverage", lambda *a, **k: 1.0)
        monkeypatch.setattr(confidence_mod, "spectral_flatness", lambda *a, **k: 5.0)
        result = confidence_gate(
            make_tone(4000, 220.0), SAMPLE_RATE, np.zeros(3), [np.zeros(3)], fake_encoder,
            enrollment_variance=np.array([0.001]), expected_active=np.ones(4000, dtype=bool),
            **self._kwargs,
        )
        assert result.accepted is False
        assert result.reason == "artifact_score"
