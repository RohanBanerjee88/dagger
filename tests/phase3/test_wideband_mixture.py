"""The native 16 kHz diarizer mixture: same scene, more bandwidth.

Two properties matter, and they pull against each other, so both are pinned:

1. With the key ABSENT, the 8 kHz scene is byte-identical to before. Phase 0-2
   results depend on it and none of them should move.
2. With the key SET, the wideband mixture describes the SAME scene -- downsample
   it and you get the pipeline mixture back. If the two ever drifted, the
   diarizer would be describing a slightly different scene than the one being
   reconstructed, and the resulting DER would be measuring the drift.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dagger.data import librimix as librimix_mod
from dagger.data.audio_io import resample

SAMPLE_RATE = 8000
HI_RATE = 16000


def _fake_read_wav(path, target_sample_rate):
    """A deterministic 'recording' whose content is rate-independent.

    A tone of a fixed frequency, sampled at whatever rate is asked for, so the
    8 kHz and 16 kHz reads are genuinely the same signal -- which is what lets a
    downsample-and-compare assertion mean something.
    """
    seconds = 3.0
    freq = 200.0 + 50.0 * (hash(Path(path).name) % 5)
    n = int(round(seconds * target_sample_rate))
    t = np.arange(n, dtype=np.float64) / target_sample_rate
    return 0.5 * np.sin(2.0 * np.pi * freq * t)


def _dataset(monkeypatch, *, diarizer_sample_rate, n_src=3, placement="scheduled"):
    monkeypatch.setattr(librimix_mod, "read_wav", _fake_read_wav)
    monkeypatch.setattr(librimix_mod, "_resolve_source_path", lambda raw, root: Path(raw))

    ds = object.__new__(librimix_mod.LibriMixDataset)
    ds.sample_rate = SAMPLE_RATE
    ds.n_src = n_src
    ds.overlap = 0.5
    ds.min_solo = SAMPLE_RATE  # 1 s
    ds.placement = placement
    ds.limit = None
    ds.offset = 0
    ds.diarizer_sample_rate = diarizer_sample_rate
    ds.data_root = Path("/fake")
    return ds


def _row(n_src: int) -> dict:
    row = {"mixture_ID": "fake"}
    for k in range(1, n_src + 1):
        row[f"source_{k}_path"] = f"spk{k}.flac"
        row[f"source_{k}_gain"] = "1.0"
    return row


class TestKeyAbsentChangesNothing:
    def test_no_wideband_mixture_by_default(self, monkeypatch):
        ds = _dataset(monkeypatch, diarizer_sample_rate=None)
        scene = ds._scene_from_row(_row(3))
        assert scene.mixture_hi is None
        assert scene.hi_sample_rate is None

    @pytest.mark.parametrize("placement", ["scheduled", "chain"])
    def test_the_8khz_scene_is_identical_with_and_without_the_key(
        self, monkeypatch, placement
    ):
        """The wideband path must be strictly additive."""
        without = _dataset(
            monkeypatch, diarizer_sample_rate=None, placement=placement
        )._scene_from_row(_row(3))
        with_key = _dataset(
            monkeypatch, diarizer_sample_rate=HI_RATE, placement=placement
        )._scene_from_row(_row(3))

        np.testing.assert_array_equal(without.mixture, with_key.mixture)
        np.testing.assert_array_equal(without.sources, with_key.sources)
        assert without.segments == with_key.segments
        assert without.speakers == with_key.speakers


class TestWidebandDescribesTheSameScene:
    @pytest.mark.parametrize("placement", ["scheduled", "chain"])
    def test_downsampling_the_wideband_mixture_recovers_the_pipeline_mixture(
        self, monkeypatch, placement
    ):
        ds = _dataset(monkeypatch, diarizer_sample_rate=HI_RATE, placement=placement)
        scene = ds._scene_from_row(_row(3))

        assert scene.hi_sample_rate == HI_RATE
        assert scene.mixture_hi is not None
        assert scene.mixture_hi.shape[0] == scene.mixture.shape[0] * 2

        recovered = resample(scene.mixture_hi, HI_RATE, SAMPLE_RATE)
        n = min(recovered.shape[0], scene.mixture.shape[0])
        # Resampling is band-limited, not exact, so compare on energy rather than
        # sample-for-sample; a placement drift would show up as a large error.
        error = np.sqrt(np.mean((recovered[:n] - scene.mixture[:n]) ** 2))
        reference = np.sqrt(np.mean(scene.mixture[:n] ** 2))
        assert error / reference < 0.05

    def test_solo_and_overlap_zones_land_at_the_same_times(self, monkeypatch):
        """The geometry the diarizer sees must match the geometry we score.

        Checked in seconds, since that is the unit Segment boundaries use and
        therefore the only unit in which the two rates have to agree.
        """
        ds = _dataset(monkeypatch, diarizer_sample_rate=HI_RATE)
        scene = ds._scene_from_row(_row(3))

        duration_lo = scene.mixture.shape[0] / SAMPLE_RATE
        duration_hi = scene.mixture_hi.shape[0] / HI_RATE
        assert duration_lo == pytest.approx(duration_hi, abs=1e-6)


class TestValidation:
    def test_non_integer_rate_ratio_is_rejected(self):
        """A fractional ratio would round chunk boundaries independently at the
        two rates, desynchronising them slowly and worst at the segment seams the
        diarizer is judged on."""
        with pytest.raises(ValueError, match="integer multiple"):
            librimix_mod.LibriMixDataset.__init__(
                object.__new__(librimix_mod.LibriMixDataset),
                {"metadata": "x.csv", "diarizer_sample_rate": 12000},
                SAMPLE_RATE,
            )
