"""The :class:`Diarizer` interface — where a real diarizer plugs into the pipeline.

Phase 0-2 never had one: every call site did ``activity_matrix(scene.segments, ...)``
directly, and ``scene.segments`` came from the clean sources
(:mod:`dagger.data.activity`), i.e. oracle diarization. Phase 3 swaps in
pyannote, so the two paths need a shared shape.

The contract is deliberately a copy of :class:`~dagger.enroll.encoder.SpeakerEncoder`,
the one abstraction in this repo that has already survived a backend swap: a tiny
ABC, one method, heavy imports deferred into method bodies so importing the
package never drags in the ``[diarize]`` extra.

``diarize`` takes the whole :class:`~dagger.data.base.Scene` rather than a bare
waveform because each backend needs a *different* view of it:
:class:`~dagger.diarize.oracle.OracleDiarizer` reads ``scene.segments``, while
:class:`~dagger.diarize.pyannote_diarizer.PyannoteDiarizer` reads
``scene.mixture_hi`` (the native-16 kHz mixture — see
:mod:`dagger.data.librimix`). Handing every backend a single canonical waveform
would force the oracle to reverse-engineer what it already knows, and would force
pyannote to run on band-limited audio.

Returned :class:`~dagger.diarize.oracle.Segment` boundaries are in **seconds**,
which is what makes the sample-rate story clean: ``activity_matrix`` multiplies
seconds by the *pipeline's* sample rate, so a 16 kHz diarizer and an 8 kHz
pipeline meet without either one knowing about the other.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    # Both imports are annotation-only, and both would close an import cycle if
    # taken at runtime: `dagger.diarize.oracle` imports this module (to subclass
    # Diarizer), and `dagger.data.base` imports Segment from that same module.
    # `from __future__ import annotations` keeps annotations lazy, so
    # TYPE_CHECKING is sufficient and this module imports nothing from the
    # package at runtime.
    from dagger.data.base import Scene
    from dagger.diarize.oracle import Segment


class DiarizationFailedError(RuntimeError):
    """A diarizer could not produce a result for this scene.

    The benign, expected-to-happen-sometimes failure — the diarization analogue
    of :class:`~dagger.enroll.topk.NoSoloRegionError`. Callers may legitimately
    catch *this specific type* to skip one arm on one scene and continue.

    The motivating case is a forced speaker count: asking for ``num_speakers=m``
    when the pipeline's internal segmentation yielded fewer embedding windows
    than ``m`` is unsatisfiable, and its clustering raises. That is a property of
    the scene being short, not a bug.

    Deliberately NOT raised for free-estimation diarization, where any failure
    means something is actually wrong and must propagate loudly rather than
    quietly shrinking the eval set.
    """


class Diarizer(abc.ABC):
    """Produce speaker-active spans for one scene.

    Implementations must be **stateless across scenes** — the eval harness runs
    several arms over the same scene and relies on each call being independent.
    """

    #: Whether this backend's segment labels ARE ``scene.speakers``, so the
    #: activity matrix should be built against that fixed label set.
    #:
    #: Only the oracle may set this. It matters in both directions:
    #:
    #: * ``True`` (oracle) — a speaker whose placement produced no segments (the
    #:   scene builders skip zero-length chunks) still gets an all-zero row, so
    #:   row ``i`` keeps meaning source ``i`` and no downstream index shifts.
    #: * ``False`` (real) — labels are discovered from the segments themselves.
    #:   Binding them to ``scene.speakers`` would make ``activity_matrix`` drop
    #:   every unrecognised cluster silently (see rule 3 in
    #:   :mod:`dagger.diarize.mapping`), hiding exactly the invented-speaker
    #:   errors Phase 3 is measuring.
    binds_scene_speakers: bool = False

    @abc.abstractmethod
    def diarize(self, scene: "Scene") -> list[Segment]:
        """Return speaker-active spans, in seconds, for ``scene``.

        Speaker labels are backend-defined and carry no relationship to
        ``scene.speakers``: a real diarizer emits anonymous cluster ids
        (``SPEAKER_00`` ...). Mapping those onto ground-truth speakers is a
        *scoring* concern handled by :mod:`dagger.diarize.mapping`, never by an
        implementation of this method.
        """
        raise NotImplementedError
