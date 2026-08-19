"""`tune_gate.py` must be able to measure V_i under REAL diarization.

This is the change that turns Phase 3's V_i claim from an inference into a
measurement, and the reason it is not a tidy-up is worth stating plainly.

Under ORACLE regions ``V_i`` is *structurally* zero: the scene scheduler gives
each speaker exactly one solo run, ``select_topk_solo_clips`` therefore returns
one clip, and ``np.var`` over a single sample is 0 by definition. The
contaminated fixture (which enrolls from a speaker's *overlap* region, where
several runs are available) is nonzero. So an oracle-only sweep compares
"identically 0" against "anything at all", separates them perfectly, and reports
a spectacular Youden's J for a property that does not exist in deployment.

Only a real diarizer -- whose fragmented solo regions yield k>1 honest clips --
produces the honest floor the contaminated population actually has to clear.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeDiarizer  # noqa: E402

from dagger.extract.base import Extractor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load_tune_gate():
    spec = importlib.util.spec_from_file_location(
        "tune_gate_phase3", ROOT / "scripts" / "tune_gate.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["tune_gate_phase3"] = module
    spec.loader.exec_module(module)
    return module


tune_gate = _load_tune_gate()


class _FakeEncoder:
    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        w = np.asarray(waveform, dtype=np.float64)
        if w.size == 0:
            return np.zeros(3)
        zcr = float(np.mean(np.abs(np.diff(np.sign(w))) > 0)) if w.size > 1 else 0.0
        return np.array([float(w.mean()), float(np.sqrt(np.mean(w**2))), zcr])


class _DeterministicExtractor(Extractor):
    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        return x_O * (0.5 + 0.25 * float(np.tanh(np.sum(embedding))))


def _measure(scene, diarizer):
    return tune_gate.measure_scene(
        scene, 0, 3, 100.0, None, _FakeEncoder(), _DeterministicExtractor(),
        diarizer=diarizer,
    )


class TestDiarizerRouting:
    def test_oracle_is_the_default_and_unchanged(self, three_speaker_scheduled_scene):
        """Every config predating the `diarizer:` block must behave identically."""
        without = tune_gate.measure_scene(
            three_speaker_scheduled_scene, 0, 3, 100.0, None,
            _FakeEncoder(), _DeterministicExtractor(),
        )
        explicit_none = _measure(three_speaker_scheduled_scene, None)

        assert len(without) == len(explicit_none)
        for a, b in zip(without, explicit_none):
            assert a["population"] == b["population"]
            assert a["speaker"] == b["speaker"]
            np.testing.assert_allclose(a["margin"], b["margin"], equal_nan=True)

    def test_a_real_diarizer_changes_the_measured_regions(
        self, three_speaker_scheduled_scene
    ):
        """If the diarizer were ignored, the whole change would be inert."""
        oracle = _measure(three_speaker_scheduled_scene, None)
        real = _measure(three_speaker_scheduled_scene, FakeDiarizer(jitter=0.2))

        def margins(rows):
            return [r["margin"] for r in rows if r["population"] == tune_gate.CORRECT]

        assert not np.allclose(
            margins(oracle), margins(real), equal_nan=True
        ), "the diarizer argument did not reach the regions"

    def test_all_four_populations_survive_real_regions(
        self, three_speaker_scheduled_scene
    ):
        """A sweep needs both labelled pairs; an empty one silently disables it.

        ``_detection_sweep`` renders "(a population is empty; nothing to sweep)"
        rather than failing, so an accidentally-empty population would produce a
        complete-looking report with one table quietly missing.
        """
        rows = _measure(three_speaker_scheduled_scene, FakeDiarizer(jitter=0.1))
        populations = {r["population"] for r in rows}
        assert {
            tune_gate.HONEST, tune_gate.CONTAMINATED,
            tune_gate.CORRECT, tune_gate.SWAPPED,
        } <= populations, populations


class TestWhyOracleVariancesAreUseless:
    def test_oracle_honest_variance_is_identically_zero(
        self, three_speaker_scheduled_scene
    ):
        """The premise of the whole change, asserted rather than asserted-in-prose.

        One solo run -> one clip -> variance over a single sample -> exactly 0.
        A sweep against this population cannot say anything about deployment.
        """
        rows = _measure(three_speaker_scheduled_scene, None)
        honest = [
            r["mean_variance"] for r in rows if r["population"] == tune_gate.HONEST
        ]
        assert honest and all(v == 0.0 for v in honest), honest

    def test_the_refusal_guard_still_fires_on_identical_populations(self):
        """`tune_gate` must refuse to recommend a threshold it cannot justify.

        Two identical distributions still produce a "best" row; copying it into
        a config would launder noise into a threshold. The guard is what makes a
        documented J ~ 0 a real negative result rather than a shrug.
        """
        rows = [
            {"population": p, "mean_variance": v, "margin": 0.0,
             "vad_coverage": 1.0, "artifact_score": 0.0}
            for p in (tune_gate.HONEST, tune_gate.CONTAMINATED)
            for v in (0.001, 0.002, 0.003)
        ]
        lines = tune_gate._detection_sweep(
            rows, "mean_variance", [0.01, 0.05],
            healthy=tune_gate.HONEST, faulty=tune_gate.CONTAMINATED,
            reject_below=False,
        )
        rendered = "\n".join(lines)
        assert "NO USABLE THRESHOLD" in rendered
        assert "suggested" not in rendered
