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

    def test_precomputed_embedding_is_used_instead_of_re_embedding(self, fake_encoder):
        # A caller that already embedded `estimate` (e.g. coarse-to-fine, which
        # re-embeds a refinement candidate before gating it) can pass that
        # embedding through and skip a redundant encoder call. Prove it's
        # actually used -- not silently ignored -- by passing a deliberately
        # wrong one and checking the margin reflects it, not a fresh embed.
        tone_a = make_tone(4000, 220.0)
        tone_b = make_tone(4000, 880.0)
        e_a = fake_encoder.embed(tone_a, SAMPLE_RATE)
        e_b = fake_encoder.embed(tone_b, SAMPLE_RATE)

        real_margin = identity_margin(tone_a, SAMPLE_RATE, e_a, [e_b], fake_encoder)
        spoofed_margin = identity_margin(
            tone_a, SAMPLE_RATE, e_a, [e_b], fake_encoder, precomputed_embedding=e_b
        )
        assert spoofed_margin != pytest.approx(real_margin)
        # cos(e_b, e_a) - cos(e_b, e_b) == cos(e_b,e_a) - 1, i.e. the "self"
        # similarity is now computed against a stand-in embedding of tone_b.
        from dagger.metrics.speaker_similarity import cosine_similarity

        expected = cosine_similarity(e_b, e_a) - cosine_similarity(e_b, e_b)
        assert spoofed_margin == pytest.approx(expected)


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

    def test_spectral_flatness_default_is_unchanged_by_the_energy_gate_parameter(self):
        # The regression lock for CLAUDE.md §7: every committed artifact_score
        # column must stay regenerable by its own generator. Passing None must
        # be the pre-2026-08-26 function exactly, not approximately.
        rng = np.random.default_rng(0)
        signal = rng.normal(size=4000) * np.repeat([1.0, 0.0], 2000)
        assert spectral_flatness(signal) == spectral_flatness(signal, min_energy_db=None)

    def test_silent_frames_inflate_ungated_flatness_and_the_gate_removes_them(self):
        # The mechanism behind the 0.005 clean-vs-G gap: a digitally silent frame
        # has every bin at the eps floor, so geo == arith and it scores exactly
        # 1.0 -- maximally "artifact-like" while containing nothing.
        tone = make_tone(4000, 220.0, amp=0.8)
        padded = np.concatenate([tone, np.zeros(4000)])
        assert spectral_flatness(padded) > spectral_flatness(tone) + 0.1
        gated = spectral_flatness(padded, min_energy_db=-40.0)
        assert abs(gated - spectral_flatness(tone, min_energy_db=-40.0)) < 0.05

    def test_energy_gated_flatness_still_separates_tone_from_noise(self):
        rng = np.random.default_rng(0)
        tone = make_tone(4000, 220.0, amp=0.8)
        noise = rng.normal(size=4000)
        assert spectral_flatness(tone, min_energy_db=-40.0) < spectral_flatness(noise, min_energy_db=-40.0)

    def test_all_silent_estimate_scores_1_in_both_modes(self):
        # Silence is a real measurement here, not an undefined one, and the two
        # modes must agree on it. If this returned nan, tune_gate.py's `_values`
        # would drop the most severe fault we can manufacture out of the sweep
        # while every table still printed a recommendation. Not exact equality:
        # geo/arith over a constant eps spectrum lands at 1.000000000000003.
        silence = np.zeros(4000)
        assert spectral_flatness(silence) == pytest.approx(1.0)
        assert spectral_flatness(silence, min_energy_db=-40.0) == pytest.approx(1.0)



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
