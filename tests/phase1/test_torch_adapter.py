"""Phase 1: fixed-length training crops on top of a SceneDataset
(dagger.data.torch_adapter).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

# `conftest.py` lives at tests/, one level above this file; its fixtures are
# auto-discovered by pytest, but plain helpers (build_staggered_scene) need an
# explicit import.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import build_staggered_scene  # noqa: E402

from dagger.data.activity import segments_from_placement  # noqa: E402
from dagger.data.base import Scene, SceneDataset  # noqa: E402
from dagger.data.torch_adapter import _overlap_runs, build_scene_crop_dataset  # noqa: E402

SAMPLE_RATE = 8000


def _scene_from_staggered(lengths, overlap, min_solo=0, name="scene"):
    built = build_staggered_scene(lengths, overlap=overlap, min_solo=min_solo, sample_rate=SAMPLE_RATE)
    segments = segments_from_placement(
        built["offsets"], built["lengths"], built["speakers"], SAMPLE_RATE
    )
    return Scene(
        mixture=built["mixture"],
        sources=built["sources"],
        segments=segments,
        speakers=built["speakers"],
        sample_rate=SAMPLE_RATE,
        name=name,
    )


class _ListSceneDataset(SceneDataset):
    def __init__(self, scenes: list[Scene]):
        self._scenes = scenes

    def __len__(self):
        return len(self._scenes)

    def __iter__(self):
        return iter(self._scenes)


class TestOverlapRuns:
    def test_single_run(self):
        overlap = np.array([0, 0, 1, 1, 1, 0, 0])
        starts, cumlen = _overlap_runs(overlap)
        np.testing.assert_array_equal(starts, [2])
        np.testing.assert_array_equal(cumlen, [3])

    def test_multiple_runs(self):
        overlap = np.array([1, 1, 0, 0, 1, 0, 1, 1, 1])
        starts, cumlen = _overlap_runs(overlap)
        np.testing.assert_array_equal(starts, [0, 4, 6])
        np.testing.assert_array_equal(cumlen, [2, 3, 6])

    def test_no_overlap_gives_empty_runs(self):
        overlap = np.zeros(10)
        starts, cumlen = _overlap_runs(overlap)
        assert starts.size == 0
        assert cumlen.size == 0

    def test_overlap_at_both_edges(self):
        overlap = np.array([1, 1, 0, 1, 1])
        starts, cumlen = _overlap_runs(overlap)
        np.testing.assert_array_equal(starts, [0, 3])
        np.testing.assert_array_equal(cumlen, [2, 4])


class TestBuildSceneCropDataset:
    def test_enrollable_scene_is_kept(self, fake_encoder):
        # enroll_speaker's default min_clip_ms is 500ms; give each speaker a
        # solo run comfortably longer than that (4000 samples @ 8kHz).
        scene = _scene_from_staggered([10000, 8000], overlap=0.3, name="ok")
        dataset = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=0.2, encoder=fake_encoder,
        )
        assert len(dataset) == 1

    def test_scene_with_no_solo_region_is_skipped(self, capsys, fake_encoder):
        # Fully overlapped (overlap=1.0) 2-speaker scene: no speaker is ever solo.
        scene = _scene_from_staggered([2000, 2000], overlap=1.0, name="no_solo")
        dataset = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=0.1, encoder=fake_encoder,
        )
        assert len(dataset) == 0
        assert "no_solo" in capsys.readouterr().out

    def test_no_encoder_means_no_embeddings_key(self):
        scene = _scene_from_staggered([4000, 3000], overlap=0.5)
        dataset = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=0.2, encoder=None,
        )
        assert len(dataset) == 1
        sample = dataset[0]
        assert "embeddings" not in sample

    def test_encoder_given_populates_embeddings_with_right_shape(self, fake_encoder):
        scene = _scene_from_staggered([10000, 8000], overlap=0.3)
        dataset = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=0.2, encoder=fake_encoder,
        )
        sample = dataset[0]
        assert sample["embeddings"].shape == (2, 3)  # FakeSpeakerEncoder emits 3-d embeddings

    def test_crop_shapes_match_segment_seconds(self):
        scene = _scene_from_staggered([4000, 3000], overlap=0.5)
        segment_seconds = 0.25
        expected_len = int(round(segment_seconds * SAMPLE_RATE))
        dataset = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=segment_seconds, encoder=None,
        )
        sample = dataset[0]
        assert sample["mixture"].shape == (expected_len,)
        assert sample["sources"].shape == (2, expected_len)
        assert sample["activity"].shape == (2, expected_len)
        assert sample["solo"].shape == (2, expected_len)
        assert sample["overlap"].shape == (expected_len,)
        assert sample["w_overlap"].shape == (2, expected_len)

    def test_crop_shorter_than_segment_is_zero_padded(self):
        scene = _scene_from_staggered([300, 300], overlap=0.5)  # short scene
        segment_seconds = 1.0  # much longer than the whole scene
        expected_len = int(round(segment_seconds * SAMPLE_RATE))
        dataset = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=segment_seconds, encoder=None,
        )
        sample = dataset[0]
        assert sample["mixture"].shape == (expected_len,)
        n = scene.mixture.shape[0]
        assert torch.all(sample["mixture"][n:] == 0.0)

    def test_require_overlap_drops_zero_overlap_scenes(self, capsys):
        # overlap=0.0 -> pure back-to-back turn-taking, no overlap anywhere.
        scene = _scene_from_staggered([2000, 2000], overlap=0.0, name="no_overlap_scene")
        kept = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=0.1, encoder=None, require_overlap=False,
        )
        dropped = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=0.1, encoder=None, require_overlap=True,
        )
        assert len(kept) == 1
        assert len(dropped) == 0
        assert "no_overlap_scene" in capsys.readouterr().out

    def test_crop_centered_on_overlap_region(self):
        """Statistically, overlap-centered crops must land inside (or very
        near) the scene's overlap run far more often than a uniform-random
        crop would for a scene where overlap is a small fraction of the
        total length."""
        scene = _scene_from_staggered([6000, 500], overlap=0.9, name="mostly_solo")
        segment_seconds = 0.03  # short crop relative to the scene
        dataset = build_scene_crop_dataset(
            _ListSceneDataset([scene]), segment_seconds=segment_seconds, encoder=None, seed=0,
        )
        seg = int(round(segment_seconds * SAMPLE_RATE))
        hits = 0
        n_draws = 50
        for _ in range(n_draws):
            sample = dataset[0]
            overlap_crop = sample["overlap"]
            if overlap_crop.sum() > 0:
                hits += 1
        # With true overlap centering essentially every draw should touch the
        # overlap region; a uniform-random start over a mostly-solo scene
        # would touch it far less often.
        assert hits / n_draws > 0.8
