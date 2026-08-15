"""A failing arm must cost that arm on that scene — never the whole scene.

Forcing `num_speakers=m` is unsatisfiable when the pipeline's segmentation yields
fewer embedding windows than `m` (sklearn: "n_samples=2 should be >= n_clusters=3").
That is a property of a short scene, not a bug, and it took down an entire real
run once.

Two wrong responses are pinned against here:

* falling back to free estimation, which would silently turn `real_forced_m`
  into a second copy of `real` and make `real_forced_m - real` read ~0 for a
  reason that has nothing to do with speaker counting;
* dropping the scene, which would cost the headline `real - oracle` comparison a
  row because a *diagnostic* arm failed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from conftest import FakeDiarizer  # noqa: E402

from dagger.diarize.base import DiarizationFailedError  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load_run_phase3():
    spec = importlib.util.spec_from_file_location(
        "run_phase3_af", ROOT / "scripts" / "run_phase3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_phase3_af"] = module
    spec.loader.exec_module(module)
    return module


run_phase3 = _load_run_phase3()

from test_run_phase3_arms import (  # noqa: E402
    GATE_CFG,
    _DeterministicExtractor,
    _FakeEncoder,
)


class _FailsWhenForced(FakeDiarizer):
    """Succeeds under free estimation, fails when handed a speaker count."""

    def __init__(self, num_speakers=None, **kw):
        super().__init__(**kw)
        self.num_speakers = num_speakers

    def diarize(self, scene):
        if self.num_speakers is not None:
            raise DiarizationFailedError(
                f"pyannote could not diarize with num_speakers={self.num_speakers}: "
                "ValueError: n_samples=2 should be >= n_clusters=3"
            )
        return super().diarize(scene)


def _run(scene, arms, monkeypatch, failures):
    monkeypatch.setattr(
        run_phase3, "_build_diarizer",
        lambda cfg, n, dev: _FailsWhenForced(num_speakers=n),
    )
    return run_phase3.score_scene_all_arms(
        scene, arms, {"name": "pyannote"}, "cpu",
        fade=0, enroll_k=3, min_clip_ms=100.0, enroll_budget_ms=None,
        encoder=_FakeEncoder(), extractor=_DeterministicExtractor(),
        gate_cfg=GATE_CFG, refine_rounds=0, diarizer_cache={},
        arm_failures=failures,
    )


ALL_ARMS = ["oracle", "real", "real_forced_m", "real_index_order"]


class TestForcedArmFailureIsContained:
    def test_the_scene_survives_for_every_other_arm(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        failures = {}
        rows, _, _ = _run(three_speaker_scheduled_scene, ALL_ARMS, monkeypatch, failures)
        present = {r["diarization"] for r in rows}
        assert "real_forced_m" not in present, "the failing arm produced rows"
        assert {"oracle", "real", "real_index_order"} <= present, \
            "a diagnostic arm's failure cost the headline comparison its rows"

    def test_the_failure_is_counted_not_swallowed(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        failures = {}
        _run(three_speaker_scheduled_scene, ALL_ARMS, monkeypatch, failures)
        assert failures == {"real_forced_m": 1}

    def test_no_silent_fallback_to_free_estimation(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        """If forced-m quietly fell back, it would duplicate `real` exactly and
        `real_forced_m - real` would read 0 dB for the wrong reason."""
        failures = {}
        rows, _, diar = _run(three_speaker_scheduled_scene, ALL_ARMS, monkeypatch, failures)
        assert not [r for r in rows if r["diarization"] == "real_forced_m"]
        assert not [r for r in diar if r["diarization"] == "real_forced_m"]

    def test_headline_pairing_is_unaffected(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        failures = {}
        rows, _, _ = _run(three_speaker_scheduled_scene, ALL_ARMS, monkeypatch, failures)

        def keyed(arm):
            return {(r["scene"], r["speaker"], r["depth"])
                    for r in rows
                    if r["diarization"] == arm and r["system"] == "no_recursion"}

        assert keyed("oracle") & keyed("real"), "real vs oracle lost its pairing"


class TestFreeEstimationFailuresStillPropagate:
    """Free estimation is not allowed to fail quietly: there is no scene-shape
    reason for it to, so a failure there means something is genuinely wrong and
    must not shrink the eval set behind our backs."""

    def test_a_free_arm_error_is_not_caught(self, three_speaker_scheduled_scene, monkeypatch):
        class _AlwaysFails(FakeDiarizer):
            def diarize(self, scene):
                raise RuntimeError("something genuinely broken")

        monkeypatch.setattr(run_phase3, "_build_diarizer", lambda cfg, n, dev: _AlwaysFails())
        with pytest.raises(RuntimeError, match="genuinely broken"):
            run_phase3.score_scene_all_arms(
                three_speaker_scheduled_scene, ["oracle", "real"],
                {"name": "pyannote"}, "cpu",
                fade=0, enroll_k=3, min_clip_ms=100.0, enroll_budget_ms=None,
                encoder=_FakeEncoder(), extractor=_DeterministicExtractor(),
                gate_cfg=GATE_CFG, refine_rounds=0, diarizer_cache={},
            )
