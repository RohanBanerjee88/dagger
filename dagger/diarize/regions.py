"""One call: scene + diarizer -> the region arrays every downstream module wants.

``activity_matrix`` -> ``solo_overlap_regions`` -> ``overlap_depth`` is spelled
out identically at six call sites today (``run_phase0/1/2.py``, ``tune_gate.py``,
``probe_phase1_conditioning.py``, ``torch_adapter.py``). Phase 3 would have made
it eight, in a script that has to run the sequence *twice per scene* under
different diarizers — so it gets a name here instead.

Deliberately additive: the existing six call sites are **left alone**. They
produce committed numbers, and a refactor that touches them would put every
Phase 0-2 result behind a "well, we did change the region code" asterisk for no
gain. New code uses this; old code keeps working.

The one behavioural subtlety lives in :func:`scene_regions`' handling of speaker
labels — see rule 3 in :mod:`dagger.diarize.mapping`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dagger.diarize.base import Diarizer
from dagger.diarize.oracle import (
    activity_matrix,
    overlap_depth,
    solo_overlap_regions,
)


@dataclass(frozen=True)
class Regions:
    """Everything the reconstruction/gate/enrollment path needs about who spoke when.

    * ``activity`` — ``a_i(t)``, ``[S, T]``, at the scene's *audio* sample rate.
    * ``speakers`` — row labels. For the oracle these are ``scene.speakers``; for
      a real diarizer they are anonymous cluster ids and row ``i`` is **not**
      ground-truth speaker ``i``.
    * ``solo`` — ``[S, T]``, region ``E_i`` (this speaker is the only active one).
    * ``overlap`` — ``[T]``, region where two or more speak and ``G`` must run.
    * ``depth`` — ``[T]``, the concurrent-speaker count ``|K|(t)``.
    """

    activity: np.ndarray
    speakers: list[str]
    solo: np.ndarray
    overlap: np.ndarray
    depth: np.ndarray

    @property
    def num_speakers(self) -> int:
        return self.activity.shape[0]


def scene_regions(scene, diarizer: Diarizer) -> Regions:
    """Run ``diarizer`` over ``scene`` and derive the region arrays.

    How speaker rows are labelled is decided by the backend, via
    :attr:`~dagger.diarize.base.Diarizer.binds_scene_speakers`, because it is the
    only party that knows whether its labels mean anything:

    * A **real** diarizer's rows are discovered from its own segments. Pinning
      them to ``scene.speakers`` would make ``activity_matrix`` silently drop
      every unrecognised cluster (see rule 3 in :mod:`dagger.diarize.mapping`),
      erasing the invented-speaker errors this phase exists to price. So the row
      count here may legitimately differ from ``len(scene.speakers)``, and
      attributing rows to ground-truth speakers is a separate, scoring-time step.
    * The **oracle**'s rows are bound to ``scene.speakers``. Its labels are the
      ground-truth ones already, and binding additionally guarantees a row per
      source even when a speaker contributed no segments at all — the scene
      builders skip zero-length chunks, and a dropped row would shift every
      downstream index while looking like a smaller scene.
    """
    activity, speakers = activity_matrix(
        diarizer.diarize(scene),
        num_samples=scene.mixture.shape[0],
        sample_rate=scene.sample_rate,
        speakers=list(scene.speakers) if diarizer.binds_scene_speakers else None,
    )
    solo, overlap = solo_overlap_regions(activity)
    return Regions(
        activity=activity,
        speakers=speakers,
        solo=solo,
        overlap=overlap,
        depth=overlap_depth(activity),
    )


def dilate_overlap(regions: Regions, samples: int) -> Regions:
    """Widen the predicted overlap region by ``samples`` on each side.

    **Why this knob exists.** Phase 3 Stage A attributed the entire -3.11 dB
    oracle-vs-real gap to the predicted activity masks, and located the mechanism
    in the DER decomposition: ``overlap_recall`` 0.758 means ~24% of true overlap
    frames are labelled non-overlap, and on those frames the pipeline takes the
    **solo-copy path** -- emitting the raw mixture verbatim as one speaker's
    track, with every other voice in it at full level.

    The two errors are wildly asymmetric, which is what makes a deliberate bias
    the right response rather than a hack:

    * calling a solo frame "overlap" is mild -- ``G`` runs where a copy would have
      sufficed, costing only whatever artifacts the extractor adds;
    * calling an overlap frame "solo" is catastrophic -- an unseparated mixture is
      emitted as a speaker's output.

    So the derived overlap mask is biased toward inclusion. ``samples`` is the
    dilation half-width; ``0`` returns ``regions`` unchanged (the identity, not a
    copy), so a sweep's zero point is exactly the undilated baseline.

    **What changes and what does not.** Only the solo/overlap *split* moves:

    * ``overlap'`` grows to ``overlap`` dilated by ``samples`` each way;
    * ``solo'_i = activity_i AND NOT overlap'``, so frames leaving solo enter the
      extract path via ``w_Oi`` (``crossfade_windows`` derives ``w_Oi`` as
      ``activity_i - w_Ei``, so the partition of unity still holds exactly);
    * ``activity`` and ``depth`` are **untouched**. ``depth`` is the
      intrinsic-difficulty diagnostic derived from ``activity`` (CLAUDE.md §6.4),
      and scoring depth comes from the reference regardless -- letting a knob on
      the audio path move the axis results are stratified by would let this
      grade itself on a curve it drew.

    ``solo'_i`` remains a subset of ``activity_i`` by construction, so
    ``enroll_speaker``'s overlap-contamination guard stays satisfied. Note the
    knob applies **uniformly**: the same ``Regions`` drives enrollment, ``x_O``
    and reconstruction. Dilation therefore shrinks the audio available to enroll
    from, and at a large enough value starves it entirely
    (``NoSoloRegionError``). That is a real cost of the knob, not a bug -- the
    sweep reports the enrollment-drop count beside the SI-SDR gain so the two are
    read together.
    """
    if samples < 0:
        raise ValueError(f"dilate_overlap needs samples >= 0, got {samples}.")
    if samples == 0:
        return regions

    # Prefix-sum sliding window rather than np.convolve: `mode="same"` returns
    # max(len(signal), len(kernel)) samples, so a dilation wider than the scene
    # silently produces an OVER-LONG mask and the broadcast against `activity`
    # blows up (or, with a squarer array, would not have). This form is O(n),
    # exact, and returns exactly `n` samples for any `samples`.
    n = regions.overlap.shape[0]
    cumulative = np.concatenate([[0.0], np.asarray(regions.overlap, dtype=np.float64)]).cumsum()
    index = np.arange(n)
    lo = np.clip(index - samples, 0, n)
    hi = np.clip(index + samples + 1, 0, n)
    overlap = ((cumulative[hi] - cumulative[lo]) > 0).astype(np.float64)
    solo = regions.activity * (1.0 - overlap)[None, :]
    return Regions(
        activity=regions.activity,
        speakers=regions.speakers,
        solo=solo,
        overlap=overlap,
        depth=regions.depth,
    )
