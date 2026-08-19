"""The dilation sweep, end to end and offline (Phase 3 Stage B item 1).

Two properties matter here and neither is visible from the unit tests on
``dilate_overlap`` itself:

1. **The diarizer is paid for ONCE per scene, however many values are swept.**
   That is the entire reason the sweep lives inside ``score_scene_all_arms``
   rather than being one invocation per value -- and it is not merely a cost
   argument. pyannote's clustering is not guaranteed bit-reproducible, so
   re-running it per sweep point could make the *regions* drift between the
   points being compared, turning a one-variable comparison into a two-variable
   one.

2. **The reported tables use the 0 ms baseline only.** Every existing Phase 3
   table (absolute SI-SDR, gap, ordering, gate) predates the sweep and does not
   filter on ``dilate_ms``. Left unfiltered, a sweep would average several
   different pipelines into one cell and label it "depth 2" -- the same
   two-variables-in-one-column mistake that cost Phase 2 five runs on the wrong
   axis.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeDiarizer  # noqa: E402

from dagger.enroll.encoder import SpeakerEncoder  # noqa: E402
from dagger.extract.base import Extractor  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load_run_phase3():
    spec = importlib.util.spec_from_file_location(
        "run_phase3_dilation", ROOT / "scripts" / "run_phase3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_phase3_dilation"] = module
    spec.loader.exec_module(module)
    return module


run_phase3 = _load_run_phase3()


class _FakeEncoder(SpeakerEncoder):
    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        w = np.asarray(waveform, dtype=np.float64)
        if w.size == 0:
            return np.zeros(3)
        zcr = float(np.mean(np.abs(np.diff(np.sign(w))) > 0)) if w.size > 1 else 0.0
        return np.array([float(w.mean()), float(np.sqrt(np.mean(w**2))), zcr])


class _DeterministicExtractor(Extractor):
    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        return x_O * (0.5 + 0.25 * float(np.tanh(np.sum(embedding))))


class _CountingDiarizer(FakeDiarizer):
    """A FakeDiarizer that records how many times it was actually invoked."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def diarize(self, scene):
        self.calls += 1
        return super().diarize(scene)


GATE_CFG = {
    "tau_margin": 0.1, "max_mean_variance": 0.05,
    "min_vad_coverage": 0.5, "max_artifact_score": 0.9,
}


def _run(scene, arms, monkeypatch, *, diarizer=None, dilate_ms_values=(0.0,),
         refine_rounds=0, refine_oracle_ceiling=False, dilation_failures=None):
    diarizer = diarizer or FakeDiarizer()
    monkeypatch.setattr(run_phase3, "_build_diarizer", lambda cfg, n, dev: diarizer)
    return run_phase3.score_scene_all_arms(
        scene, arms, {"name": "pyannote"}, "cpu",
        fade=0, enroll_k=3, min_clip_ms=100.0, enroll_budget_ms=None,
        encoder=_FakeEncoder(), extractor=_DeterministicExtractor(),
        gate_cfg=GATE_CFG, refine_rounds=refine_rounds, diarizer_cache={},
        dilate_ms_values=list(dilate_ms_values),
        refine_oracle_ceiling=refine_oracle_ceiling,
        dilation_failures=dilation_failures,
    )


class TestTheDiarizerIsPaidForOnce:
    def test_sweeping_many_values_does_not_rerun_the_diarizer(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        """Six sweep points must cost exactly one diarization, not six."""
        diarizer = _CountingDiarizer()
        _run(
            three_speaker_scheduled_scene, ["oracle", "real"], monkeypatch,
            diarizer=diarizer, dilate_ms_values=(0.0, 10.0, 25.0, 50.0, 100.0, 200.0),
        )
        assert diarizer.calls == 1, (
            f"real diarizer ran {diarizer.calls} times for one scene -- the sweep "
            "is re-running it per value, so the regions can drift between the "
            "points being compared"
        )

    def test_arms_sharing_regions_still_share_them(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        """`real` and `real_index_order` differ only in order, so one call."""
        diarizer = _CountingDiarizer()
        _run(
            three_speaker_scheduled_scene, ["oracle", "real", "real_index_order"],
            monkeypatch, diarizer=diarizer, dilate_ms_values=(0.0, 50.0),
        )
        assert diarizer.calls == 1


class TestRowsCarryTheirSweepPoint:
    def test_every_row_is_stamped(self, three_speaker_scheduled_scene, monkeypatch):
        values = (0.0, 25.0, 100.0)
        rows, gate_rows, diar_rows = _run(
            three_speaker_scheduled_scene, ["oracle"], monkeypatch,
            dilate_ms_values=values, refine_rounds=1,
        )
        for payload in (rows, gate_rows, diar_rows):
            assert {r["dilate_ms"] for r in payload} == set(values)

    def test_dilation_changes_the_scores(self, three_speaker_scheduled_scene, monkeypatch):
        """A sweep whose points are identical would be measuring nothing."""
        rows, _, _ = _run(
            three_speaker_scheduled_scene, ["oracle"], monkeypatch,
            dilate_ms_values=(0.0, 200.0),
        )

        def scores(value):
            return {
                (r["speaker"], r["system"], r["depth"]): r["si_sdr"]
                for r in rows if r["dilate_ms"] == value
            }

        base, widened = scores(0.0), scores(200.0)
        shared = set(base) & set(widened)
        assert shared, "no comparable rows across sweep points"
        assert any(
            not np.isclose(base[k], widened[k], equal_nan=True) for k in shared
        ), "dilation left every score untouched"

    def test_der_is_identical_across_sweep_points(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        """Dilation must not move DER -- it does not touch `activity`.

        If it did, the knob would be grading its own homework: an audio-path
        choice would change the diarization-quality number used to explain it.
        """
        _, _, diar_rows = _run(
            three_speaker_scheduled_scene, ["oracle", "real"], monkeypatch,
            diarizer=FakeDiarizer(jitter=0.1), dilate_ms_values=(0.0, 150.0),
        )
        for arm in ("oracle", "real"):
            ders = {
                r["dilate_ms"]: r["der"]
                for r in diar_rows if r["diarization"] == arm
            }
            assert ders[0.0] == ders[150.0], arm

    def test_mask_recall_rises_with_dilation(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        """The quantity the knob targets, and the cost it pays for it."""
        _, _, diar_rows = _run(
            three_speaker_scheduled_scene, ["oracle", "real"], monkeypatch,
            diarizer=FakeDiarizer(jitter=0.2), dilate_ms_values=(0.0, 200.0),
        )
        real = {r["dilate_ms"]: r for r in diar_rows if r["diarization"] == "real"}
        assert real[200.0]["mask_overlap_recall"] >= real[0.0]["mask_overlap_recall"]
        assert (
            real[200.0]["mask_overlap_false_alarm"]
            >= real[0.0]["mask_overlap_false_alarm"]
        )


class TestAggressiveDilationDoesNotPoisonTheBaseline:
    def test_an_unenrollable_sweep_point_is_counted_not_propagated(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        """A dilation big enough to erase every solo region must not kill the scene.

        Letting it propagate would drop the scene from EVERY sweep point --
        including 0 ms, the one point that has to stay comparable with the
        committed Stage A numbers. The failure is recorded so the knob's cost
        appears in the report rather than as a quietly shrinking `n`.
        """
        failures: dict = {}
        scene = three_speaker_scheduled_scene
        huge = 1000.0 * scene.mixture.shape[0] / scene.sample_rate  # whole scene

        rows, _, _ = _run(
            scene, ["oracle"], monkeypatch,
            dilate_ms_values=(0.0, huge), dilation_failures=failures,
        )

        assert any(r["dilate_ms"] == 0.0 for r in rows), (
            "the baseline sweep point was lost to a failure at another point"
        )
        assert not any(r["dilate_ms"] == huge for r in rows)
        assert failures.get(("oracle", huge)) == 1


class TestReportsUseTheBaselineOnly:
    def test_absolute_table_does_not_average_across_sweep_points(self, tmp_path):
        """Mixing dilation values into one cell would relabel a mean silently."""
        def row(dilate_ms, si_sdr):
            return {
                "diarization": "oracle", "dilate_ms": dilate_ms, "scene": "s",
                "speaker": "s1", "system": "no_recursion", "m": 3, "depth": 2,
                "si_sdr": si_sdr, "deflation_index": None,
                "n_accepted_before": None, "refine_rounds": 0,
                "cluster": "s1", "n_clusters": 3,
            }

        rows = [row(0.0, 10.0), row(50.0, -10.0)]
        run_phase3._write_results(
            rows, [], [], tmp_path, "stem", ["oracle"], n_scenes=1,
        )
        text = (tmp_path / "stem.md").read_text()

        # The baseline value, not the mean of the two sweep points (0.00).
        assert "| oracle | no_recursion | 10.00 |" in text
        assert "0 ms baseline only" in text
