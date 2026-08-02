"""Tests for the ``offset`` config key on both dataset loaders (Phase 2).

``limit`` alone is a head-slice, so every config drawing from the same metadata
CSV sees the same leading rows -- there was no way to carve a dev split that is
provably disjoint from a training slice. Selecting gate thresholds on the test
scenes and then reporting results on those same scenes is leakage (CLAUDE.md
§8), so threshold tuning needs one.

These tests exercise the row-selection logic only (no audio is read), following
tests/phase2/test_librimix_scheduled.py's pattern of writing a small metadata
file under a fake data root.
"""

from __future__ import annotations

import pytest

import dagger.data.librimix as librimix_mod
import dagger.data.wsj0mix as wsj0mix_mod

SAMPLE_RATE = 8000


def _write_librimix_metadata(tmp_path, n_rows: int):
    header = "mixture_ID,source_1_path,source_1_gain,source_2_path,source_2_gain"
    rows = [f"mix{i},a{i},1.0,b{i},1.0" for i in range(n_rows)]
    (tmp_path / "meta.csv").write_text("\n".join([header, *rows]) + "\n")


def _librimix(tmp_path, monkeypatch, **cfg):
    monkeypatch.setattr(librimix_mod, "resolve_data_root", lambda: tmp_path)
    return librimix_mod.LibriMixDataset({"metadata": "meta.csv", **cfg}, SAMPLE_RATE)


def _wsj0mix(tmp_path, monkeypatch, n_rows: int, **cfg):
    (tmp_path / "list.txt").write_text("\n".join(f"utt{i}" for i in range(n_rows)) + "\n")
    monkeypatch.setattr(wsj0mix_mod, "resolve_data_root", lambda: tmp_path)
    monkeypatch.setattr(wsj0mix_mod, "ensure_access", lambda root: None)
    return wsj0mix_mod.Wsj0MixDataset({"metadata": "list.txt", **cfg}, SAMPLE_RATE)


class TestLibriMixOffset:
    def test_defaults_to_zero_so_existing_configs_are_unchanged(self, tmp_path, monkeypatch):
        _write_librimix_metadata(tmp_path, 10)

        ds = _librimix(tmp_path, monkeypatch, limit=3)

        assert ds.offset == 0
        assert [r["mixture_ID"] for r in ds.rows] == ["mix0", "mix1", "mix2"]

    def test_offset_skips_rows_before_limit_applies(self, tmp_path, monkeypatch):
        _write_librimix_metadata(tmp_path, 10)

        ds = _librimix(tmp_path, monkeypatch, offset=4, limit=3)

        assert [r["mixture_ID"] for r in ds.rows] == ["mix4", "mix5", "mix6"]

    def test_a_training_slice_and_a_dev_split_share_no_scenes(self, tmp_path, monkeypatch):
        """The property the key exists for: a run with `limit: N` and a run with
        `offset: N` are disjoint by construction, no bookkeeping required."""
        _write_librimix_metadata(tmp_path, 20)

        train = _librimix(tmp_path, monkeypatch, limit=12)
        dev = _librimix(tmp_path, monkeypatch, offset=12, limit=5)

        train_ids = {r["mixture_ID"] for r in train.rows}
        dev_ids = {r["mixture_ID"] for r in dev.rows}
        assert train_ids & dev_ids == set()
        assert len(dev_ids) == 5

    def test_offset_without_limit_takes_the_rest(self, tmp_path, monkeypatch):
        _write_librimix_metadata(tmp_path, 6)

        ds = _librimix(tmp_path, monkeypatch, offset=4)

        assert [r["mixture_ID"] for r in ds.rows] == ["mix4", "mix5"]

    def test_offset_past_the_end_yields_nothing_rather_than_wrapping(self, tmp_path, monkeypatch):
        _write_librimix_metadata(tmp_path, 3)

        ds = _librimix(tmp_path, monkeypatch, offset=99, limit=5)

        assert len(ds) == 0

    def test_negative_offset_is_rejected(self, tmp_path, monkeypatch):
        """A negative offset would silently index from the end -- taking the
        tail of the file while looking like a small skip."""
        _write_librimix_metadata(tmp_path, 5)

        with pytest.raises(ValueError, match="offset must be >= 0"):
            _librimix(tmp_path, monkeypatch, offset=-2)


class TestWsj0MixOffset:
    def test_defaults_to_zero_so_existing_configs_are_unchanged(self, tmp_path, monkeypatch):
        ds = _wsj0mix(tmp_path, monkeypatch, 10, limit=3)

        assert ds.offset == 0
        assert ds.lines == ["utt0", "utt1", "utt2"]

    def test_offset_skips_rows_before_limit_applies(self, tmp_path, monkeypatch):
        ds = _wsj0mix(tmp_path, monkeypatch, 10, offset=4, limit=3)

        assert ds.lines == ["utt4", "utt5", "utt6"]

    def test_negative_offset_is_rejected(self, tmp_path, monkeypatch):
        with pytest.raises(ValueError, match="offset must be >= 0"):
            _wsj0mix(tmp_path, monkeypatch, 5, offset=-1)
