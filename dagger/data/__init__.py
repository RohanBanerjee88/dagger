"""Dataset loaders that feed the Phase 0 plumbing path.

Every loader yields a :class:`~dagger.data.base.Scene` — the exact bundle
``scripts/run_phase0.py`` used to synthesize by hand (mixture, per-speaker clean
sources, oracle segments, speaker order, sample rate) — so the downstream
pipeline (``activity_matrix`` -> ``solo_overlap_regions`` -> ``overlap_mixture``
-> ``reconstruct_all`` -> SI-SDR) is reused verbatim.

Design notes (CLAUDE.md §5, §7):

* **Storage-lean.** Real corpora (WSJ0-2mix, LibriMix) are *mixed on the fly*
  from source utterances, so only the sources live on disk — never the mixtures.
* **Remote compute.** Audio lives on a mounted volume resolved via the
  ``DAGGER_DATA_ROOT`` env var (see :mod:`dagger.data.paths`); nothing is
  committed to the repo.
* **numpy-only.** Phase 0 stays pure numpy (soundfile + scipy for I/O); torch
  arrives with the extractor ``G`` in Phase 1.
"""

from dagger.data.base import Scene, SceneDataset, build_dataset

__all__ = ["Scene", "SceneDataset", "build_dataset"]
