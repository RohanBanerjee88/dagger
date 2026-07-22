"""Tests for LibriMixDataset's ``placement: scheduled`` config key (Phase 2).

Builds a ``LibriMixDataset`` instance without touching real files: ``read_wav``
is monkeypatched to synthesize deterministic tones from the row, matching the
pattern of the rest of the suite (numpy-only synthetic scenes, see
``tests/conftest.py``). This exercises the actual dispatch in
``LibriMixDataset._scene_from_row`` -- not just the underlying mixing/activity
functions (already covered by ``tests/phase2/test_scheduler.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

import dagger.data.librimix as librimix_mod
from dagger.diarize.oracle import activity_matrix, overlap_depth, solo_overlap_regions

SAMPLE_RATE = 8000


def _make_dataset(monkeypatch, *, n_src: int, overlap: float, min_solo_ms: float, placement: str, lengths: list[int]):
    monkeypatch.setattr(librimix_mod, "resolve_data_root", lambda: __import__("pathlib").Path("/fake"))

    def fake_read_wav(path, target_sample_rate):
        # path encodes which speaker/row via "s{k}" written by the fake row below.
        k = int(str(path).split("s")[-1])
        length = lengths[k - 1]
        t = np.arange(length, dtype=np.float64) / SAMPLE_RATE
        return 0.5 * np.sin(2.0 * np.pi * (220.0 * k) * t)

    monkeypatch.setattr(librimix_mod, "read_wav", fake_read_wav)

    ds = object.__new__(librimix_mod.LibriMixDataset)
    ds.sample_rate = SAMPLE_RATE
    ds.n_src = n_src
    ds.overlap = overlap
    ds.min_solo = int(round(min_solo_ms / 1000.0 * SAMPLE_RATE))
    ds.placement = placement
    ds.limit = None
    ds.data_root = __import__("pathlib").Path("/fake")
    return ds


def _fake_row(n_src: int) -> dict:
    row = {"mixture_ID": "fake"}
    for k in range(1, n_src + 1):
        row[f"source_{k}_path"] = f"s{k}"
        row[f"source_{k}_gain"] = "1.0"
    return row


class TestPlacementConfigDispatch:
    def test_rejects_unknown_placement(self, monkeypatch, tmp_path):
        (tmp_path / "meta.csv").write_text("mixture_ID\n")
        monkeypatch.setattr(librimix_mod, "resolve_data_root", lambda: tmp_path)
        with pytest.raises(ValueError, match="placement"):
            librimix_mod.LibriMixDataset(
                {"metadata": "meta.csv", "placement": "bogus"}, SAMPLE_RATE
            )

    def test_default_placement_is_chain(self, monkeypatch, tmp_path):
        (tmp_path / "meta.csv").write_text("mixture_ID,source_1_path,source_1_gain,source_2_path,source_2_gain\nx,a,1.0,b,1.0\n")
        monkeypatch.setattr(librimix_mod, "resolve_data_root", lambda: tmp_path)
        ds = librimix_mod.LibriMixDataset({"metadata": "meta.csv"}, SAMPLE_RATE)
        assert ds.placement == "chain"

    def test_scheduled_placement_produces_guaranteed_solo_and_depth(self, monkeypatch):
        ds = _make_dataset(
            monkeypatch, n_src=3, overlap=0.5, min_solo_ms=100.0, placement="scheduled",
            lengths=[4000, 4000, 4000],
        )
        scene = ds._scene_from_row(_fake_row(3))

        n = scene.mixture.shape[0]
        activity, speakers = activity_matrix(
            scene.segments, num_samples=n, sample_rate=SAMPLE_RATE, speakers=scene.speakers
        )
        solo, _overlap = solo_overlap_regions(activity)
        depth = overlap_depth(activity)

        for i in range(3):
            assert solo[i].sum() > 0  # every speaker gets guaranteed solo time
        assert depth.max() == 3  # and the scene reaches a genuine depth-3 overlap

    def test_chain_placement_unchanged_for_scheduled_config_absent(self, monkeypatch):
        ds_chain = _make_dataset(
            monkeypatch, n_src=2, overlap=0.5, min_solo_ms=0.0, placement="chain",
            lengths=[4000, 3000],
        )
        scene = ds_chain._scene_from_row(_fake_row(2))
        # chain offsets place s2 partway into s1 (overlap=0.5); total length is
        # less than the two lengths summed, i.e. they really do overlap.
        assert scene.mixture.shape[0] < 4000 + 3000
