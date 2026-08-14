"""End-to-end arm behaviour, offline: no pyannote, no GPU, no corpus.

`score_scene_all_arms` is where a plumbing mistake would be least visible and
most damaging — an arm quietly scoring the wrong regions produces a perfectly
plausible gap table. These tests exercise the whole path with a fake diarizer,
a fake encoder and a deterministic extractor.
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
    """Load the script by path, as the Phase 2 reporting tests do."""
    spec = importlib.util.spec_from_file_location(
        "run_phase3", ROOT / "scripts" / "run_phase3.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_phase3"] = module
    spec.loader.exec_module(module)
    return module


run_phase3 = _load_run_phase3()


class _FakeEncoder(SpeakerEncoder):
    """Cheap, deterministic 3-d embedding: [mean, rms, zero-crossing rate].

    Mirrors ``tests/conftest.py``'s FakeSpeakerEncoder so the embedding actually
    varies with the audio -- a constant embedding would make the conditioning
    path untestable.
    """

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        w = np.asarray(waveform, dtype=np.float64)
        if w.size == 0:
            return np.zeros(3)
        zcr = float(np.mean(np.abs(np.diff(np.sign(w))) > 0)) if w.size > 1 else 0.0
        return np.array([float(w.mean()), float(np.sqrt(np.mean(w**2))), zcr])


class _DeterministicExtractor(Extractor):
    """Scales x_O by a stable function of the embedding.

    Deterministic so repeated arms are comparable, and embedding-dependent so
    the deflation order genuinely changes the outputs (an extractor that ignored
    its embedding would make the order-policy test vacuous).
    """

    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        weight = 0.5 + 0.25 * float(np.tanh(np.sum(embedding)))
        return x_O * weight


GATE_CFG = {
    "tau_margin": 0.1,
    "max_mean_variance": 0.05,
    "min_vad_coverage": 0.5,
    "max_artifact_score": 0.9,
}


def _run(scene, arms, monkeypatch, *, diarizer=None, refine_rounds=0):
    """Drive score_scene_all_arms with a fake diarizer in place of pyannote."""
    diarizer = diarizer or FakeDiarizer()
    monkeypatch.setattr(run_phase3, "_build_diarizer", lambda cfg, n, dev: diarizer)
    return run_phase3.score_scene_all_arms(
        scene, arms, {"name": "pyannote"}, "cpu",
        fade=0, enroll_k=3, min_clip_ms=100.0, enroll_budget_ms=None,
        encoder=_FakeEncoder(), extractor=_DeterministicExtractor(),
        gate_cfg=GATE_CFG, refine_rounds=refine_rounds, diarizer_cache={},
    )


class TestArmsProduceDistinctLabelledRows:
    def test_every_arm_is_tagged_and_present(self, three_speaker_scheduled_scene, monkeypatch):
        arms = ["oracle", "real", "real_index_order"]
        rows, gate_rows, diar_rows = _run(three_speaker_scheduled_scene, arms, monkeypatch)

        assert {r["diarization"] for r in rows} == set(arms)
        assert {r["diarization"] for r in gate_rows} == set(arms)
        assert {r["diarization"] for r in diar_rows} == set(arms)

    def test_oracle_arm_scores_a_perfect_der(self, three_speaker_scheduled_scene, monkeypatch):
        _, _, diar_rows = _run(three_speaker_scheduled_scene, ["oracle"], monkeypatch)
        assert diar_rows[0]["der"] == 0.0
        assert diar_rows[0]["n_missed_speakers"] == 0

    def test_real_arm_scores_a_nonzero_der(self, three_speaker_scheduled_scene, monkeypatch):
        _, _, diar_rows = _run(
            three_speaker_scheduled_scene, ["oracle", "real"], monkeypatch,
            diarizer=FakeDiarizer(jitter=0.25),
        )
        real = next(r for r in diar_rows if r["diarization"] == "real")
        assert real["der"] > 0.0


class TestOrderPolicyIsIsolated:
    """`real` and `real_index_order` differ ONLY in deflation order.

    So the two systems that do not depend on that order must be bit-identical
    across them. If they are not, the order policy has leaked into something it
    should not touch -- and the whole decomposition
    (`real - real_index_order` = "the reordering V_i induces") would be measuring
    that leak instead.
    """

    @pytest.mark.parametrize("system", ["no_recursion", "coarse_to_fine"])
    def test_order_independent_systems_are_identical_across_the_two_arms(
        self, three_speaker_scheduled_scene, monkeypatch, system
    ):
        rows, _, _ = _run(
            three_speaker_scheduled_scene, ["oracle", "real", "real_index_order"], monkeypatch
        )

        def scores(arm):
            return {
                (r["speaker"], r["depth"]): r["si_sdr"]
                for r in rows
                if r["diarization"] == arm and r["system"] == system
            }

        left, right = scores("real"), scores("real_index_order")
        assert left and left.keys() == right.keys()
        for key in left:
            assert left[key] == right[key] or (
                np.isnan(left[key]) and np.isnan(right[key])
            ), f"{system} differs at {key} between real and real_index_order"

    def test_the_two_arms_share_one_diarizer_pass(self, three_speaker_scheduled_scene, monkeypatch):
        """Regions must be computed once and reused, not re-derived per arm.

        A real backend's clustering is not guaranteed bit-reproducible, so
        re-running it would let the two arms differ in their REGIONS as well as
        their order -- destroying the one-variable-at-a-time property.
        """
        calls = {"n": 0}
        inner = FakeDiarizer()

        class CountingDiarizer(FakeDiarizer):
            def diarize(self, scene):
                calls["n"] += 1
                return inner.diarize(scene)

        _run(
            three_speaker_scheduled_scene,
            ["oracle", "real", "real_index_order"],
            monkeypatch,
            diarizer=CountingDiarizer(),
        )
        assert calls["n"] == 1


class TestCardinalityFailuresSurviveToTheReport:
    def test_a_dropped_speaker_is_reported_not_hidden(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        _, _, diar_rows = _run(
            three_speaker_scheduled_scene, ["oracle", "real"], monkeypatch,
            diarizer=FakeDiarizer(drop_speaker=1),
        )
        real = next(r for r in diar_rows if r["diarization"] == "real")
        assert real["n_pred"] == 2
        assert real["n_missed_speakers"] == 1

    def test_an_invented_speaker_is_reported_not_hidden(
        self, three_speaker_scheduled_scene, monkeypatch
    ):
        rows, _, diar_rows = _run(
            three_speaker_scheduled_scene, ["oracle", "real"], monkeypatch,
            diarizer=FakeDiarizer(extra_cluster=True),
        )
        real = next(r for r in diar_rows if r["diarization"] == "real")
        assert real["n_pred"] == 4
        assert real["n_spurious_clusters"] == 1
        # The phantom cluster has no ground-truth target, so it contributes no
        # score rows -- it must not be scored against some other speaker's source.
        assert all(r["speaker"] != "SPEAKER_99" for r in rows)


class TestMandatoryOracleArm:
    def test_config_without_oracle_is_rejected(self, tmp_path, monkeypatch):
        """Guardrail §6.2, enforced rather than documented."""
        import yaml

        cfg = {
            "sample_rate": 8000,
            "dataset": {"name": "librimix", "metadata": "x.csv", "n_src": 3},
            "diarizer": {"arms": ["real"]},
        }
        path = tmp_path / "no_oracle.yaml"
        path.write_text(yaml.safe_dump(cfg))
        monkeypatch.setattr(sys, "argv", ["run_phase3.py", "--config", str(path)])
        with pytest.raises(SystemExit, match="oracle"):
            run_phase3.main()
