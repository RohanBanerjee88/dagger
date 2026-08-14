"""Map a real diarizer's anonymous clusters onto ground-truth speakers, for SCORING ONLY.

A real diarizer emits ``SPEAKER_00``, ``SPEAKER_01`` ... in an order that has no
relationship to ``Scene.sources`` row order. Everything downstream of the oracle
path assumed row ``i`` *is* ground-truth speaker ``i`` — reconstruction indexes
outputs by row, and Phase 2's scoring compares ``outputs[i]`` against
``scene.sources[i]`` with no permutation search anywhere. Feed that a permuted
activity matrix and nothing raises; it simply reports wrong numbers that look
plausible. This module is the fix, and it is governed by three rules.

**1. The mapping is scoring-only.** It decides which ground-truth source an
output row is *compared against*. It never reorders enrollment, never selects a
deflation order, never touches ``x_O`` or anything the extractor sees. The audio
path runs entirely on predicted rows and never learns the true labels — which is
the point, since at inference on real data there are no true labels. A test
asserts this separation.

**2. ``k != m`` is a result, not an error.** A diarizer that finds 2 clusters for
3 speakers has *missed a speaker*; one that finds 4 has *invented* one. Both are
real diarization failures and both are exactly what Phase 3 exists to price.
Dropping either from the table would understate the oracle-vs-real gap, so
unmatched entities are reported on both sides rather than discarded.

**3. Never resolve clusters through ``activity_matrix(speakers=...)``.**
:func:`~dagger.diarize.oracle.activity_matrix` *silently skips* segments whose
speaker is not in the ``speakers`` list it was given. Passing ``scene.speakers``
to bind predicted labels would therefore make every extra cluster vanish without
a trace — turning rule 2's "invented a speaker" into "no evidence of a problem."
The real arm must call it with ``speakers=None`` so clusters are discovered, then
map afterwards.

The assignment itself is the standard optimal-mapping step used by every DER
implementation: maximise total frame agreement between predicted and reference
rows via the Hungarian algorithm. Nothing novel, and deliberately so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusterMapping:
    """The scoring-time correspondence between predicted rows and reference rows.

    * ``cluster_to_ref[c]`` — reference row that predicted row ``c`` was matched
      to, or ``None`` if the assignment left it unmatched (a *spurious* cluster).
    * ``ref_to_cluster[r]`` — the inverse, ``None`` for a reference speaker the
      diarizer never found (a *missed* speaker).
    * ``frame_agreement`` — ``[n_pred, n_ref]`` count of samples where both rows
      are active. The raw cost matrix, kept so callers can report *how* confident
      a match was rather than only that one happened.
    """

    cluster_to_ref: tuple[int | None, ...]
    ref_to_cluster: tuple[int | None, ...]
    frame_agreement: np.ndarray

    @property
    def unmatched_clusters(self) -> tuple[int, ...]:
        """Predicted rows with no reference counterpart (invented speakers)."""
        return tuple(c for c, r in enumerate(self.cluster_to_ref) if r is None)

    @property
    def unmatched_refs(self) -> tuple[int, ...]:
        """Reference speakers the diarizer never found (missed speakers)."""
        return tuple(r for r, c in enumerate(self.ref_to_cluster) if c is None)


def _hungarian(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximising assignment on ``cost``; scipy if present, brute force if not.

    scipy arrives with the ``[data]`` extra, which the numpy-only core install
    deliberately does not require. The fallback is exact (it enumerates
    permutations) and only ever runs on the tiny matrices this project produces —
    speaker counts here are 2-5, and it is capped so a surprise never turns into
    a hang.
    """
    try:
        from scipy.optimize import linear_sum_assignment

        return linear_sum_assignment(cost, maximize=True)
    except ImportError:
        pass

    from itertools import permutations

    n_rows, n_cols = cost.shape
    size = max(n_rows, n_cols)
    if size > 8:
        raise ImportError(
            "dagger.diarize.mapping needs scipy for assignments this large "
            f"({n_rows}x{n_cols}); install the '[data]' extra."
        )
    # Pad to square with zero-gain cells so a single permutation sweep covers
    # every injective assignment; padded cells contribute nothing to the score
    # and are dropped from the result below.
    padded = np.zeros((size, size), dtype=cost.dtype)
    padded[:n_rows, :n_cols] = cost
    best_score, best_perm = -np.inf, tuple(range(size))
    for perm in permutations(range(size)):
        score = float(sum(padded[r, perm[r]] for r in range(size)))
        if score > best_score:
            best_score, best_perm = score, perm
    pairs = [(r, best_perm[r]) for r in range(n_rows) if best_perm[r] < n_cols]
    return (
        np.asarray([r for r, _ in pairs], dtype=int),
        np.asarray([c for _, c in pairs], dtype=int),
    )


def map_clusters_to_speakers(
    pred_activity: np.ndarray,
    ref_activity: np.ndarray,
    *,
    min_agreement: int = 1,
) -> ClusterMapping:
    """Optimally match predicted activity rows to reference activity rows.

    Both are ``[rows, T]`` binary masks at the *same* sample rate and length —
    the caller is responsible for that, and a length mismatch raises rather than
    broadcasting into a silently wrong overlap count.

    ``min_agreement`` drops matches that share fewer than this many active
    samples. The Hungarian step will otherwise happily pair two rows with **zero**
    frames in common purely to complete the assignment, which reports a
    correspondence where the evidence is that there is none; such a pair is
    better described as one missed speaker plus one spurious cluster.
    """
    pred_activity = np.asarray(pred_activity)
    ref_activity = np.asarray(ref_activity)
    if pred_activity.ndim != 2 or ref_activity.ndim != 2:
        raise ValueError(
            "map_clusters_to_speakers expects [rows, T] activity matrices, got "
            f"{pred_activity.shape} and {ref_activity.shape}."
        )
    if pred_activity.shape[1] != ref_activity.shape[1]:
        raise ValueError(
            "predicted and reference activity must cover the same number of "
            f"samples, got {pred_activity.shape[1]} vs {ref_activity.shape[1]}."
        )

    pred_bool = pred_activity.astype(bool)
    ref_bool = ref_activity.astype(bool)
    agreement = (pred_bool.astype(np.int64) @ ref_bool.T.astype(np.int64))

    n_pred, n_ref = agreement.shape
    cluster_to_ref: list[int | None] = [None] * n_pred
    ref_to_cluster: list[int | None] = [None] * n_ref

    if n_pred and n_ref:
        rows, cols = _hungarian(agreement)
        for c, r in zip(rows.tolist(), cols.tolist()):
            if agreement[c, r] < min_agreement:
                continue  # no shared speech: a miss and a spurious cluster, not a match
            cluster_to_ref[c] = r
            ref_to_cluster[r] = c

    return ClusterMapping(
        cluster_to_ref=tuple(cluster_to_ref),
        ref_to_cluster=tuple(ref_to_cluster),
        frame_agreement=agreement,
    )
