"""Tests for scripts/train_phase1.py's multi-depth curriculum training support
(CLAUDE.md Phase 2 "Stage 2"): `train.proposed`'s `dataset:` config may be a
*list* of per-depth entries instead of one dict, so one training run can draw
batches from several overlap depths. `build_dataset` is faked here (returns
pre-built scenes keyed by `n_src`) so this test needs no real LibriMix
metadata/audio, and `TitaNetEncoder` is swapped for the suite's existing
`FakeSpeakerEncoder` so it needs no GPU/model download.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import build_staggered_scene  # noqa: E402

from dagger.data.activity import segments_from_placement  # noqa: E402
from dagger.data.base import Scene, SceneDataset  # noqa: E402

SAMPLE_RATE = 8000


def _load_train_phase1():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "train_phase1_under_test", repo_root / "scripts" / "train_phase1.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train_phase1 = _load_train_phase1()


class _ListSceneDataset(SceneDataset):
    def __init__(self, scenes: list[Scene]):
        self._scenes = scenes

    def __len__(self) -> int:
        return len(self._scenes)

    def __iter__(self):
        return iter(self._scenes)


def _scene(lengths: list[int], name: str) -> Scene:
    # min_solo=4000 samples (500ms @ 8kHz) guarantees every speaker a solo
    # window comfortably above the test's enroll.min_clip_ms=400.0 below.
    built = build_staggered_scene(lengths, overlap=0.3, min_solo=4000, sample_rate=SAMPLE_RATE)
    segments = segments_from_placement(built["offsets"], built["lengths"], built["speakers"], SAMPLE_RATE)
    return Scene(
        mixture=built["mixture"], sources=built["sources"], segments=segments,
        speakers=built["speakers"], sample_rate=SAMPLE_RATE, name=name,
    )


TINY_EXTRACTOR = dict(
    hidden_channels=8, n_blocks=1, n_fft=64, hop_length=16, n_heads=2,
    embed_dim=3, n_tokens=2, cross_attn_blocks=1,  # embed_dim=3 matches FakeSpeakerEncoder
)


def _base_cfg(tmp_path, dataset_cfg, checkpoint_stem: str) -> dict:
    return {
        "sample_rate": SAMPLE_RATE,
        "fade_ms": 5,
        "dataset": dataset_cfg,
        "enroll": {"k": 3, "min_clip_ms": 400.0},
        "extractor": TINY_EXTRACTOR,
        "train": {
            "system": "proposed",
            "epochs": 1,
            "batch_size": 1,
            "segment_seconds": 0.3,
            "lr": 1e-3,
            "checkpoint_out": str(tmp_path / f"{checkpoint_stem}.pt"),
        },
    }


class TestCurriculumTraining:
    def test_multi_depth_dataset_list_trains_and_records_both_depths(
        self, monkeypatch, tmp_path, fake_encoder,
    ):
        scenes_by_n_src = {
            2: [_scene([12000, 10000], "two")],
            3: [_scene([12000, 10000, 11000], "three")],
        }

        def fake_build_dataset(cfg):
            return _ListSceneDataset(scenes_by_n_src[cfg["dataset"]["n_src"]])

        monkeypatch.setattr(train_phase1, "build_dataset", fake_build_dataset)
        monkeypatch.setattr(
            "dagger.enroll.encoder.TitaNetEncoder", lambda device="cpu": fake_encoder
        )

        cfg = _base_cfg(
            tmp_path,
            dataset_cfg=[{"name": "librimix", "n_src": 2}, {"name": "librimix", "n_src": 3}],
            checkpoint_stem="curriculum",
        )

        train_phase1.train_proposed(cfg, device="cpu")

        ckpt_path = tmp_path / "proposed_curriculum.pt"
        assert ckpt_path.is_file()
        state = torch.load(ckpt_path, map_location="cpu")
        assert sorted(state["trained_n_src"]) == [2, 3]

    def test_single_dataset_dict_is_unaffected(self, monkeypatch, tmp_path, fake_encoder):
        """Backward-compat: a bare dict under `dataset:` (every pre-existing
        config's shape) must behave exactly as before -- one loader, one
        depth, `trained_n_src` a single-element list."""
        scenes = [_scene([12000, 10000], "only")]
        monkeypatch.setattr(train_phase1, "build_dataset", lambda cfg: _ListSceneDataset(scenes))
        monkeypatch.setattr(
            "dagger.enroll.encoder.TitaNetEncoder", lambda device="cpu": fake_encoder
        )

        cfg = _base_cfg(
            tmp_path, dataset_cfg={"name": "librimix", "n_src": 2}, checkpoint_stem="single",
        )

        train_phase1.train_proposed(cfg, device="cpu")

        ckpt_path = tmp_path / "proposed_single.pt"
        assert ckpt_path.is_file()
        state = torch.load(ckpt_path, map_location="cpu")
        assert state["trained_n_src"] == [2]
