"""The :class:`Scene` bundle and the :class:`SceneDataset` interface.

A :class:`Scene` is exactly what the Phase 0 pipeline needs: the one-channel
mixture, the per-speaker clean sources (for scoring), the oracle segments, the
speaker order, and the sample rate. Both backends — LibriMix and WSJ0-2mix —
produce this same object, so the pipeline downstream of the loader is identical
regardless of corpus.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from dagger.diarize.oracle import Segment


@dataclass(frozen=True)
class Scene:
    """One mixture and everything needed to run + score the Phase 0 path.

    * ``mixture``  — 1-D one-channel mixture ``x`` of shape ``[T]``.
    * ``sources``  — clean per-speaker sources ``[S, T]``; row ``i`` aligns to
      ``speakers[i]`` and is the ground-truth target for SI-SDR.
    * ``segments`` — oracle speaker-active spans (derived from the clean
      sources); feed :func:`dagger.diarize.oracle.activity_matrix`.
    * ``speakers`` — stable speaker order; row ``i`` of ``sources``/activity.
    * ``sample_rate`` — Hz; masks are sampled at this rate so they align with
      the waveform exactly (Phase 0 red flag: framerate/sample-rate mismatch).
    """

    mixture: np.ndarray
    sources: np.ndarray
    segments: list[Segment]
    speakers: list[str]
    sample_rate: int
    #: Optional human-readable id for logging (e.g. the mixture id).
    name: str = ""


class SceneDataset(abc.ABC):
    """A lazily-iterable collection of :class:`Scene` objects.

    Backends stream one scene into memory at a time (corpora are large; on-the-fly
    mixing keeps only the current item's sources resident). ``len()`` reports how
    many scenes the (optionally ``limit``-capped) dataset will yield.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __iter__(self):
        raise NotImplementedError


def build_dataset(cfg: dict) -> SceneDataset:
    """Construct a :class:`SceneDataset` from a Phase 0 config.

    Dispatches on ``cfg["dataset"]["name"]``: ``librimix`` | ``wsj0mix``. Both
    read audio from the mounted volume (``DAGGER_DATA_ROOT``) and mix on the fly.
    """
    sample_rate = int(cfg["sample_rate"])
    ds = cfg["dataset"]
    name = str(ds["name"]).lower()

    if name == "librimix":
        from dagger.data.librimix import LibriMixDataset

        return LibriMixDataset(ds, sample_rate)
    if name == "wsj0mix":
        from dagger.data.wsj0mix import Wsj0MixDataset

        return Wsj0MixDataset(ds, sample_rate)
    raise ValueError(
        f"Unknown dataset name {name!r}; expected 'librimix' or 'wsj0mix'."
    )
