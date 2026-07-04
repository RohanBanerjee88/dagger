"""LibriMix loader: mix on the fly from LibriSpeech sources (CLAUDE.md §5).

LibriMix mixtures are sums of clean single-speaker LibriSpeech utterances. Rather
than store the (large) pre-generated mixture wavs, we read the source utterances
listed in a LibriMix metadata CSV and mix them here — so only LibriSpeech lives
on the mounted volume.

Phase 0 mixes *staggered* (see :mod:`dagger.data.mixing`) so each scene has solo
and overlap regions. Oracle segments come from the clean sources
(:mod:`dagger.data.activity`), not from an RTTM.

Config block (``cfg["dataset"]``):

    name: librimix
    metadata: metadata/Libri2Mix/libri2mix_dev-clean.csv   # under DAGGER_DATA_ROOT
    n_src: 2                # 2 or 3
    overlap: 0.5            # staggered-overlap fraction (mixing.stagger_offsets)
    limit: 8               # cap the number of scenes (a handful for Phase 0)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dagger.data.activity import segments_from_sources
from dagger.data.audio_io import read_wav
from dagger.data.base import Scene, SceneDataset
from dagger.data.mixing import mix_sources, stagger_offsets
from dagger.data.paths import resolve_data_root


def _resolve_source_path(raw: str, data_root: Path) -> Path:
    """Re-root a CSV source path onto ``data_root``.

    LibriMix metadata often stores absolute paths from the machine that generated
    it. We try, in order: the path as-is (if it exists), ``data_root / raw``, and
    finally re-rooting from the ``LibriSpeech`` component onward — so metadata
    generated elsewhere still resolves against the local mount.
    """
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    joined = data_root / raw
    if joined.exists():
        return joined
    parts = p.parts
    if "LibriSpeech" in parts:
        idx = parts.index("LibriSpeech")
        return data_root.joinpath(*parts[idx:])
    return joined  # let the reader raise a clear FileNotFoundError


class LibriMixDataset(SceneDataset):
    """Streams :class:`Scene` objects mixed on the fly from LibriMix metadata."""

    def __init__(self, cfg: dict, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self.n_src = int(cfg.get("n_src", 2))
        self.overlap = float(cfg.get("overlap", 0.5))
        self.limit = cfg.get("limit")
        self.data_root = resolve_data_root()
        self.metadata = self.data_root / str(cfg["metadata"])
        if not self.metadata.is_file():
            raise FileNotFoundError(
                f"LibriMix metadata CSV not found at {self.metadata!r}."
            )
        self.rows = self._read_rows()

    def _read_rows(self) -> list[dict]:
        with open(self.metadata, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if self.limit is not None:
            rows = rows[: int(self.limit)]
        return rows

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        for row in self.rows:
            yield self._scene_from_row(row)

    def _scene_from_row(self, row: dict) -> Scene:
        speakers = [f"s{k}" for k in range(1, self.n_src + 1)]
        sources_raw: list[np.ndarray] = []
        gains: list[float] = []
        for k in range(1, self.n_src + 1):
            path = _resolve_source_path(row[f"source_{k}_path"], self.data_root)
            sources_raw.append(read_wav(path, self.sample_rate))
            gains.append(float(row.get(f"source_{k}_gain", 1.0)))

        offsets = stagger_offsets([len(s) for s in sources_raw], self.overlap)
        sources, mixture = mix_sources(
            sources_raw, gains=gains, offsets=offsets, length_mode="max"
        )
        segments = segments_from_sources(sources, speakers, self.sample_rate)
        return Scene(
            mixture=mixture,
            sources=sources,
            segments=segments,
            speakers=speakers,
            sample_rate=self.sample_rate,
            name=str(row.get("mixture_ID", "")),
        )
