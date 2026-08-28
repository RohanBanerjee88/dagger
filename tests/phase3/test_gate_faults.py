"""The manufactured gate-fault fixtures (dagger/gate/faults.py, NOT DEPLOYABLE).

Every test here exists because of a specific way this project has been fooled
before. The two that matter most are the ones that would go red if a corruption
were replaced by the identity function -- CLAUDE.md §9: "a test that cannot fail
is not a passing test", which shipped twice.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.audio.provenance import original_mixture
from dagger.gate.artifact import spectral_flatness, vad_coverage
from dagger.gate.faults import add_noise, attenuate, drop_span, punch_holes
from dagger.reconstruct.stitch import crossfade_windows

SAMPLE_RATE = 8000


def _speechlike(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SAMPLE_RATE
    harmonics = sum(np.sin(2 * np.pi * 130.0 * k * t) / k for k in range(1, 16))
    return harmonics / np.abs(harmonics).max() + 0.001 * rng.normal(size=n)


def _region(n: int, start: int, stop: int) -> np.ndarray:
    region = np.zeros(n, dtype=bool)
    region[start:stop] = True
    return region


class TestFaultsTouchOnlyTheRegion:
    @pytest.mark.parametrize("corrupt", [
        lambda s, r: drop_span(s, r, 0.5),
        lambda s, r: attenuate(s, r, -30.0),
        lambda s, r: add_noise(s, r, 0.0),
        lambda s, r: punch_holes(s, r, 0.75),
    ])
    def test_samples_outside_the_region_are_untouched(self, corrupt):
        # The contract that makes "50% dropped" mean half of the region G is
        # responsible for, rather than half of a track that is mostly other
        # speakers' turns.
        signal = _speechlike(8000)
        region = _region(8000, 2000, 6000)
        out = corrupt(signal, region)
        assert np.array_equal(out[~region], signal[~region])


class TestVadFaultsAreDetectable:
    def test_dropout_lowers_vad_coverage_monotonically(self):
        signal = _speechlike(8000)
        region = _region(8000, 0, 8000)
        coverages = [
            vad_coverage(drop_span(signal, region, f, rng=np.random.default_rng(0)),
                         region, SAMPLE_RATE)
            for f in (0.0, 0.25, 0.5, 0.75)
        ]
        # Graded, not binary: a fixture that jumps straight to 0 scores J = 1.0
        # at every candidate threshold and therefore places none.
        assert all(a >= b for a, b in zip(coverages, coverages[1:]))
        assert coverages[0] - coverages[-1] > 0.3

    def test_attenuation_is_detectable_only_below_active_masks_floor(self):
            # Two facts in one test. (1) A whole-signal gain is invisible: active_mask
            # thresholds against the CLIP'S OWN peak -- the same scale-invariance that
            # hid Q5's 2.86x level error from every SI-SDR here. (2) A region-selective
            # gain is visible, but ONLY below the -40 dB floor. Pinning both is what
            # stops a `quiet_*` fixture from silently going inert again.
            signal = _speechlike(8000)
            region = _region(8000, 4000, 8000)
            baseline = vad_coverage(signal, region, SAMPLE_RATE)
            assert vad_coverage(attenuate(signal, region, -30.0), region, SAMPLE_RATE) == pytest.approx(baseline)
            assert vad_coverage(attenuate(signal, region, -50.0), region, SAMPLE_RATE) < baseline - 0.5
            uniform = attenuate(signal, np.ones(8000, dtype=bool), -50.0)
            assert vad_coverage(uniform, region, SAMPLE_RATE) == pytest.approx(baseline)


class TestArtifactFaults:
    def test_additive_noise_raises_flatness_monotonically(self):
        signal = _speechlike(8000)
        region = _region(8000, 0, 8000)
        scores = [
            spectral_flatness(add_noise(signal, region, snr, rng=np.random.default_rng(0)),
                              min_energy_db=-40.0)
            for snr in (40.0, 20.0, 10.0, 0.0)
        ]
        assert all(a <= b for a, b in zip(scores, scores[1:]))
        assert scores[-1] - scores[0] > 0.1

    def test_punch_holes_is_exact_resynthesis_when_nothing_is_punched(self):
        # Makes any measured effect attributable to the punching rather than to
        # the STFT round-trip. Without this the fixture and the transform are
        # confounded and neither can be blamed.
        signal = _speechlike(8000)
        region = _region(8000, 1000, 7000)
        out = punch_holes(signal, region, 0.0)
        assert np.allclose(out, signal, atol=1e-8)

    def test_punch_holes_actually_changes_the_signal(self):
        signal = _speechlike(8000)
        region = _region(8000, 1000, 7000)
        out = punch_holes(signal, region, 0.75, rng=np.random.default_rng(0))
        assert not np.allclose(out[region], signal[region], atol=1e-6)


class TestStitchEquivalence:
    def test_stitch_matches_reconstruct_all(self):
        # Guards the tune_gate.py refactor that lifts G's forward pass out of
        # reconstruct_all so one extraction can feed ten fault populations. If
        # this drifts, the healthy `correct` population silently stops being the
        # thing every prior run measured -- and nothing else would notice.
        from dagger.extract.base import Extractor
        from dagger.reconstruct.stitch import reconstruct_all

        class _Gain(Extractor):
            def _extract(self, x_O, embedding):
                return x_O * float(embedding[0])

        n = 4000
        rng = np.random.default_rng(0)
        mixture = rng.normal(size=n)
        activity = np.zeros((2, n)); activity[0, :3000] = 1.0; activity[1, 1000:] = 1.0
        solo = np.zeros((2, n)); solo[0, :1000] = 1.0; solo[1, 3000:] = 1.0
        overlap = ((activity[0] > 0) & (activity[1] > 0)).astype(np.float64)
        x = original_mixture(mixture, label="x")
        x_O = original_mixture(mixture * overlap, label="x_O")
        embeddings = np.array([[0.5], [0.25]])
        extractor = _Gain()
        fade = 40

        expected = reconstruct_all(x, x_O, activity, solo, embeddings, extractor, fade=fade)
        x_samples = np.asarray(x, dtype=np.float64)
        for i in range(2):
            g_out = np.asarray(extractor.extract(x_O, embeddings[i]), dtype=np.float64)
            w_Ei, w_Oi = crossfade_windows(solo[i], activity[i], fade=fade)
            assert np.array_equal(x_samples * w_Ei + g_out * w_Oi, expected[i])


class TestFaultsStayOutOfTheEvalPath:
    def test_no_eval_module_imports_the_fault_fixtures(self):
        # Same idea as the no-residual-in-audio-path guard: an invariant that is
        # obvious today, and a tripwire for the day someone reaches for a
        # convenient corruption helper from inside a reported system.
        from pathlib import Path

        root = Path(__file__).resolve().parents[2]
        eval_path = [
            root / "dagger" / "eval" / "systems.py",
            root / "dagger" / "refine" / "coarse_to_fine.py",
            root / "dagger" / "reconstruct" / "deflation.py",
            root / "dagger" / "gate" / "__init__.py",
            root / "scripts" / "run_phase2.py",
            root / "scripts" / "run_phase3.py",
        ]
        offenders = [p.name for p in eval_path if p.exists() and "gate.faults" in p.read_text()]
        assert not offenders, f"fault fixtures are NOT DEPLOYABLE but are imported by {offenders}"
