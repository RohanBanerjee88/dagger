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
