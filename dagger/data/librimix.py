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
    placement: chain        # "chain" (default, Phase 0/1) or "scheduled" (Phase 2:
                             # guaranteed per-speaker solo + a synchronized deep-
                             # overlap zone -- see mixing.schedule_solo_then_overlap)
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from dagger.data.activity import segments_from_chunks, segments_from_placement
from dagger.data.audio_io import read_wav
from dagger.data.base import Scene, SceneDataset
from dagger.data.mixing import (
    mix_scheduled_sources,
    mix_sources,
    schedule_solo_then_overlap,
    stagger_offsets,
)
from dagger.data.paths import resolve_data_root


def _resolve_source_path(raw: str, data_root: Path) -> Path:
    """Re-root a CSV source path onto ``data_root``.

    LibriMix metadata comes in two shapes depending on how it was generated:
    absolute paths from the machine that built it (containing a ``LibriSpeech``
    component we can re-root from), or paths already relative to the
    LibriSpeech root itself (e.g. ``dev-clean/1272/...``, the shape the
    official generator's checked-in CSVs use). We try, in order: the path
    as-is (if absolute and it exists), ``data_root / raw``, re-rooting from the
    ``LibriSpeech`` component onward, and finally nesting under
    ``data_root / "LibriSpeech"`` for the relative-to-root shape.
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
        rerooted = data_root.joinpath(*parts[idx:])
        if rerooted.exists():
            return rerooted
    under_librispeech = data_root / "LibriSpeech" / raw
    if under_librispeech.exists():
        return under_librispeech
    return joined  # let the reader raise a clear FileNotFoundError


class LibriMixDataset(SceneDataset):
    """Streams :class:`Scene` objects mixed on the fly from LibriMix metadata."""

    def __init__(self, cfg: dict, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self.n_src = int(cfg.get("n_src", 2))
        self.overlap = float(cfg.get("overlap", 0.5))
        # Guaranteed per-speaker solo window (see stagger_offsets). Default 1 s:
        # comfortably above enrollment's min_clip_ms=500 default, so 3-mix scenes
        # stop being skipped for lack of solo audio.
        self.min_solo = int(round(float(cfg.get("min_solo_ms", 1000.0)) / 1000.0 * self.sample_rate))
        self.placement = str(cfg.get("placement", "chain"))
        if self.placement not in ("chain", "scheduled"):
            raise ValueError(f"placement must be 'chain' or 'scheduled', got {self.placement!r}.")
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

        lengths = [len(s) for s in sources_raw]
        if self.placement == "scheduled":
            chunks = schedule_solo_then_overlap(lengths, min_solo=self.min_solo)
            sources, mixture = mix_scheduled_sources(
                sources_raw, chunks, gains=gains, length_mode="max"
            )
            segments = segments_from_chunks(chunks, speakers, self.sample_rate)
        else:
            offsets = stagger_offsets(lengths, self.overlap, min_solo=self.min_solo)
            sources, mixture = mix_sources(
                sources_raw, gains=gains, offsets=offsets, length_mode="max"
            )
            segments = segments_from_placement(offsets, lengths, speakers, self.sample_rate)
        return Scene(
            mixture=mixture,
            sources=sources,
            segments=segments,
            speakers=speakers,
            sample_rate=self.sample_rate,
            name=str(row.get("mixture_ID", "")),
        )
