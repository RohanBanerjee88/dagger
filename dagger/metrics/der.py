"""Diarization error rate, computed frame-wise on activity matrices.

Phase 3 needs to say *how wrong* the diarizer was, not just that quality dropped
— guardrail §6.2's whole point is attribution. DER is the standard number, and
this computes it directly on the ``[S, T]`` masks the pipeline already builds
rather than round-tripping through ``pyannote.core.Annotation``. That keeps the
metric available with no ``[diarize]`` extra installed, and guarantees it scores
*exactly* the masks reconstruction consumed rather than an equivalent-looking
re-derivation.

**Components are reported separately, never just the total.** CLAUDE.md §6.4
forbids reporting an aggregate average as evidence, and DER is a sum of three
unlike failures:

* **miss** — reference speech nobody was assigned to. Overlap is where this
  concentrates, since a diarizer that reports one speaker in a 3-way overlap
  misses two speakers' worth of frames.
* **false alarm** — predicted speech where the reference has none.
* **confusion** — both agree someone is speaking, but the count assigned to the
  *matched* speakers falls short, i.e. the speech went to the wrong identity.

``overlap_recall`` is broken out for a reason specific to this project: it is the
fraction of true overlap frames the diarizer also called overlap, and overlap is
precisely where ``G`` runs at all. Solo frames are copied straight through
(CLAUDE.md §2), so a diarization error there costs almost nothing, while an
overlap frame the diarizer thinks is solo silently removes a speaker from the
extractor's job. Expect this to track our SI-SDR gap far better than total DER.

The speaker correspondence comes from :mod:`dagger.diarize.mapping` — the same
optimal mapping used for scoring, so the DER and the SI-SDR table can never
disagree about who was who.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dagger.diarize.mapping import ClusterMapping, map_clusters_to_speakers


@dataclass(frozen=True)
class DERResult:
    """DER and its decomposition, all in *samples* except the ratios.

    ``der == (miss + false_alarm + confusion) / total_speech``, the NIST
    definition. It can exceed 1.0 — false alarms are unbounded — which is correct
    and not a bug to clamp away.
    """

    der: float
    miss: int
    false_alarm: int
    confusion: int
    total_speech: int
    overlap_recall: float
    n_ref: int
    n_pred: int
    n_missed_speakers: int
    n_spurious_clusters: int

    @property
    def miss_rate(self) -> float:
        return self.miss / self.total_speech if self.total_speech else float("nan")

    @property
    def false_alarm_rate(self) -> float:
        return self.false_alarm / self.total_speech if self.total_speech else float("nan")

    @property
    def confusion_rate(self) -> float:
        return self.confusion / self.total_speech if self.total_speech else float("nan")


def diarization_error_rate(
    ref_activity: np.ndarray,
    pred_activity: np.ndarray,
    *,
    mapping: ClusterMapping | None = None,
) -> DERResult:
    """Frame-wise DER of ``pred_activity`` against ``ref_activity``.

    Both are ``[rows, T]`` binary masks at the same rate and length. Rows need not
    correspond: ``mapping`` (or a fresh optimal one) decides that.

    This is the *overlap-aware* formulation, which is the only correct one here —
    scoring per-frame speaker **counts** rather than a single label per frame.
    At each sample:

    * ``ref_n`` = reference speakers active, ``pred_n`` = predicted speakers active
    * ``correct`` = how many *matched* speaker pairs are active in both
    * miss ``+= max(ref_n - pred_n, 0)``, false alarm ``+= max(pred_n - ref_n, 0)``
    * confusion ``+= min(ref_n, pred_n) - correct``

    A single-label-per-frame DER would score every overlap frame as at best
    partially right by construction, which for a project *about* overlap would
    make the metric mostly measure its own simplification.
    """
    ref_activity = np.asarray(ref_activity)
    pred_activity = np.asarray(pred_activity)
    if ref_activity.ndim != 2 or pred_activity.ndim != 2:
        raise ValueError(
            "diarization_error_rate expects [rows, T] activity matrices, got "
            f"{ref_activity.shape} and {pred_activity.shape}."
        )
    if ref_activity.shape[1] != pred_activity.shape[1]:
        raise ValueError(
            "reference and predicted activity must cover the same number of "
            f"samples, got {ref_activity.shape[1]} vs {pred_activity.shape[1]}."
        )

    ref = ref_activity.astype(bool)
    pred = pred_activity.astype(bool)
    if mapping is None:
        mapping = map_clusters_to_speakers(pred, ref)

    ref_n = ref.sum(axis=0).astype(np.int64)
    pred_n = pred.sum(axis=0).astype(np.int64)

    # Only MATCHED pairs can be "correct": an unmatched cluster's frames are, by
    # definition, not attributable to any reference speaker.
    correct = np.zeros(ref.shape[1], dtype=np.int64)
    for cluster, reference in enumerate(mapping.cluster_to_ref):
        if reference is None:
            continue
        correct += (pred[cluster] & ref[reference]).astype(np.int64)

    miss = int(np.maximum(ref_n - pred_n, 0).sum())
    false_alarm = int(np.maximum(pred_n - ref_n, 0).sum())
    confusion = int((np.minimum(ref_n, pred_n) - correct).sum())
    total_speech = int(ref_n.sum())

    ref_overlap = ref_n >= 2
    pred_overlap = pred_n >= 2
    n_ref_overlap = int(ref_overlap.sum())
    overlap_recall = (
        float((ref_overlap & pred_overlap).sum()) / n_ref_overlap
        if n_ref_overlap
        else float("nan")
    )

    return DERResult(
        der=(miss + false_alarm + confusion) / total_speech if total_speech else float("nan"),
        miss=miss,
        false_alarm=false_alarm,
        confusion=confusion,
        total_speech=total_speech,
        overlap_recall=overlap_recall,
        n_ref=int(ref.shape[0]),
        n_pred=int(pred.shape[0]),
        n_missed_speakers=len(mapping.unmatched_refs),
        n_spurious_clusters=len(mapping.unmatched_clusters),
    )
