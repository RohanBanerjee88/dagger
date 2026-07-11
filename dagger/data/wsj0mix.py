"""WSJ0-2mix loader: mix on the fly from WSJ0 sources (CLAUDE.md §5).

Like LibriMix, a WSJ0-2mix mixture is a sum of clean single-speaker WSJ0
utterances scaled to a relative loudness — here the per-utterance SNRs listed in
the standard ``mix_2_spk_{cv,tt}.txt`` file. We read the sources and mix them
here, so only WSJ0 lives on the mounted volume, never the mixtures.

Access note (honest): WSJ0 has no API key — it is gated by a signed LDC license
and a manual corpus download. :func:`ensure_access` reads
``DAGGER_WSJ0_ACCESS_KEY`` as the credential to authorize a *fetch* when the
licensed data is hosted behind a private endpoint; when the data already sits on
``DAGGER_DATA_ROOT`` the key is unused. This wires the credential in without
pretending WSJ0 authenticates in a way it does not.

Config block (``cfg["dataset"]``):

    name: wsj0mix
    metadata: metadata/wsj0-2mix/mix_2_spk_cv.txt   # under DAGGER_DATA_ROOT
    n_src: 2
    overlap: 0.5
    limit: 8
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dagger.data.activity import segments_from_placement
from dagger.data.audio_io import read_wav
from dagger.data.base import Scene, SceneDataset
from dagger.data.mixing import db_to_linear, mix_sources, stagger_offsets
from dagger.data.paths import get_credential, resolve_data_root

_CREDENTIAL_ENV = "DAGGER_WSJ0_ACCESS_KEY"


def ensure_access(data_root: Path) -> None:
    """Authorize access to the (LDC-gated) WSJ0 data.

    If WSJ0 audio is already present under ``data_root`` this is a no-op. If it is
    absent, we require ``DAGGER_WSJ0_ACCESS_KEY`` to be set — the credential a
    private mirror / gated endpoint would use to serve the licensed data — and
    raise an actionable error otherwise. (Fetching from that endpoint is left to
    the deployment; this hook is the single authorization choke point.)
    """
    if (data_root / "wsj0").exists() or any(data_root.glob("**/si_*")):
        return  # data already on the mounted volume; no credential needed
    if get_credential(_CREDENTIAL_ENV) is None:
        raise PermissionError(
            "WSJ0 audio was not found under DAGGER_DATA_ROOT and "
            f"{_CREDENTIAL_ENV} is not set. WSJ0 is LDC-licensed: either mount the "
            f"corpus on the volume, or set {_CREDENTIAL_ENV} to authorize a fetch "
            "from your private mirror. See .env.example."
        )


def _parse_line(line: str) -> list[tuple[str, float]]:
    """Parse one ``mix_2_spk`` line into ``[(source_path, snr_db), ...]``.

    Format is whitespace-separated ``path1 snr1 path2 snr2 [...]`` — one
    (path, SNR) pair per source.
    """
    toks = line.split()
    if len(toks) % 2 != 0:
        raise ValueError(f"Malformed mix line (expected path/SNR pairs): {line!r}")
    return [(toks[i], float(toks[i + 1])) for i in range(0, len(toks), 2)]


class Wsj0MixDataset(SceneDataset):
    """Streams :class:`Scene` objects mixed on the fly from a WSJ0-2mix list."""

    def __init__(self, cfg: dict, sample_rate: int):
        self.sample_rate = int(sample_rate)
        self.n_src = int(cfg.get("n_src", 2))
        self.overlap = float(cfg.get("overlap", 0.5))
        # Guaranteed per-speaker solo window (see stagger_offsets); same default
        # rationale as LibriMixDataset.
        self.min_solo = int(round(float(cfg.get("min_solo_ms", 1000.0)) / 1000.0 * self.sample_rate))
        self.limit = cfg.get("limit")
        self.data_root = resolve_data_root()
        ensure_access(self.data_root)

        self.metadata = self.data_root / str(cfg["metadata"])
        if not self.metadata.is_file():
            raise FileNotFoundError(
                f"WSJ0-2mix list not found at {self.metadata!r}."
            )
        self.lines = self._read_lines()

    def _read_lines(self) -> list[str]:
        text = self.metadata.read_text(encoding="utf-8").splitlines()
        lines = [ln.strip() for ln in text if ln.strip()]
        if self.limit is not None:
            lines = lines[: int(self.limit)]
        return lines

    def __len__(self) -> int:
        return len(self.lines)

    def __iter__(self):
        for i, line in enumerate(self.lines):
            yield self._scene_from_line(i, line)

    def _scene_from_line(self, index: int, line: str) -> Scene:
        pairs = _parse_line(line)[: self.n_src]
        speakers = [f"s{k}" for k in range(1, len(pairs) + 1)]
        sources_raw: list[np.ndarray] = []
        gains: list[float] = []
        for path, snr_db in pairs:
            resolved = path if Path(path).is_absolute() else str(self.data_root / path)
            sources_raw.append(read_wav(resolved, self.sample_rate))
            gains.append(db_to_linear(snr_db))

        lengths = [len(s) for s in sources_raw]
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
            name=f"mix_{index:04d}",
        )
