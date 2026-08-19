"""Simulated diarization error for extractor training (Phase 3 Stage B item 7).

``G`` has only ever been trained on **oracle** activity masks, and Phase 3
Stage A priced what that costs: -3.11 dB at depth 2 under real diarization, with
the loss attributed entirely to the predicted masks rather than to the deflation
order. Training against corrupted masks is the direct remedy.

The corruption must match the MEASURED error profile, not an assumed one
---------------------------------------------------------------------------
This is the part worth reading before changing anything here. The augmentation
originally planned for this phase was a mix of label swaps and boundary jitter,
designed off the short-scene runs where DER decomposed as confusion 0.171-0.187.
Those runs turned out to be broken for an unrelated reason (pyannote returned
~2 clusters for 3 speakers at 10-20 s scene length), and at the 2-minute
operating point the profile is completely different:

    miss 0.105  |  false alarm 0.000  |  confusion 0.008  |  overlap_recall 0.758

Confusion has essentially vanished; **miss dominates**. So label-swap
augmentation would train ``G`` against a failure that does not occur, and would
spend capacity doing it. What ~24% missed overlap recall actually means is that
a speaker who IS talking inside an overlapped region gets marked inactive --
which drops the region's depth, and sends those frames down the solo-copy path
where the raw mixture is emitted verbatim.

So that is what this module simulates: **dropped speaker activity inside
overlapped regions**. Not boundary jitter, not label swaps.

Note the asymmetry with :func:`dagger.diarize.regions.dilate_overlap`, which
attacks the same measured defect from the inference side by biasing the overlap
mask toward inclusion. The two are complementary and independent: dilation makes
the deployed pipeline conservative, augmentation makes ``G`` robust when it is
not conservative enough.
"""

from __future__ import annotations

import numpy as np


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous ``(start, end)`` runs where ``mask`` is truthy."""
    padded = np.concatenate([[0], np.asarray(mask).astype(np.int8) > 0, [0]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def drop_overlapped_activity(
    activity: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    drop_prob: float = 0.25,
    min_dur_ms: float = 200.0,
    max_dur_ms: float = 2000.0,
) -> np.ndarray:
    """Return a copy of ``activity`` with spans deleted inside overlapped regions.

    For each speaker, every contiguous run where that speaker is active **and at
    least one other speaker is also active** is a candidate. With probability
    ``drop_prob`` a candidate run has a random sub-span of
    ``[min_dur_ms, max_dur_ms]`` zeroed, reproducing a diarizer that failed to
    notice this speaker during an overlap.

    Two deliberate constraints:

    * **Only overlapped runs are touched.** Dropping a speaker's *solo* activity
      would delete them from the scene entirely rather than mislabel them, which
      is a different (and unmeasured) failure -- and it would starve enrollment,
      confounding the augmentation with a change in embedding quality.
    * **A speaker is never fully deleted.** At least one overlapped run per
      speaker survives untouched, so ``activity`` never collapses to all-zero
      for a row. A zero row propagates into ``crossfade_windows`` as a
      degenerate partition and into enrollment as a hard failure, which would
      show up as scenes silently vanishing from the training set -- the same
      selection bias that shrank Phase 1's effective dataset.

    ``rng`` is passed in rather than created here so the caller owns
    reproducibility (``build_scene_crop_dataset`` seeds one generator for the
    whole preparation pass).
    """
    if not 0.0 <= drop_prob <= 1.0:
        raise ValueError(f"drop_prob must be in [0, 1], got {drop_prob}.")
    if min_dur_ms > max_dur_ms:
        raise ValueError("min_dur_ms must not exceed max_dur_ms.")

    activity = np.asarray(activity, dtype=np.float64)
    if drop_prob == 0.0:
        return activity.copy()

    augmented = activity.copy()
    depth = activity.sum(axis=0)
    min_samples = max(1, int(round(min_dur_ms / 1000.0 * sample_rate)))
    max_samples = max(min_samples, int(round(max_dur_ms / 1000.0 * sample_rate)))

    for i in range(activity.shape[0]):
        overlapped = (activity[i] > 0) & (depth >= 2)
        candidates = [(s, e) for s, e in _runs(overlapped) if (e - s) >= min_samples]
        if len(candidates) < 2:
            # Fewer than two candidates means dropping the only one would erase
            # this speaker's entire overlap participation. Leave the row alone.
            continue
        # Reserve one run so the speaker always survives somewhere.
        keep = int(rng.integers(0, len(candidates)))
        for index, (start, end) in enumerate(candidates):
            if index == keep or rng.random() >= drop_prob:
                continue
            span = int(rng.integers(min_samples, min(max_samples, end - start) + 1))
            offset = int(rng.integers(0, (end - start) - span + 1))
            augmented[i, start + offset: start + offset + span] = 0.0

    return augmented
