"""Unwrapping the pyannote pipeline's return value, across API shapes.

pyannote 3.x returned a bare ``Annotation``; 4.x returns a ``DiarizeOutput``
wrapper, so a fixed ``output.itertracks(...)`` raises ``AttributeError`` — which
is exactly what the first real Phase 3 run hit. These fakes pin every shape so a
version bump surfaces here rather than at the top of a GPU session.

The `exclusive_speaker_diarization` test is the important one: picking that
attribute would produce *numbers rather than a crash*, and they would be wrong
in the one way this project cannot tolerate.
"""

from __future__ import annotations

import pytest

from dagger.diarize.pyannote_diarizer import _to_segments


class _Turn:
    def __init__(self, start: float, end: float):
        self.start, self.end = start, end


class _Annotation:
    """Stands in for pyannote.core.Annotation (3.x shape)."""

    def __init__(self, triples):
        self._triples = triples

    def itertracks(self, yield_label=False):
        assert yield_label, "dagger always asks for labels"
        return iter(self._triples)


class _DiarizeOutput:
    """Stands in for pyannote 4.x's DiarizeOutput wrapper."""

    def __init__(self, speaker_diarization, exclusive=None):
        self.speaker_diarization = speaker_diarization
        self.exclusive_speaker_diarization = exclusive


OVERLAPPING = [
    (_Turn(0.0, 1.0), "_", "SPEAKER_00"),
    (_Turn(0.5, 1.5), "_", "SPEAKER_01"),  # overlaps the first
]


class TestApiShapes:
    def test_bare_annotation_3x(self):
        segs = _to_segments(_Annotation(OVERLAPPING))
        assert [s.speaker for s in segs] == ["SPEAKER_00", "SPEAKER_01"]
        assert segs[0].start == 0.0 and segs[0].duration == pytest.approx(1.0)

    def test_diarize_output_wrapper_4x(self):
        segs = _to_segments(_DiarizeOutput(_Annotation(OVERLAPPING)))
        assert [s.speaker for s in segs] == ["SPEAKER_00", "SPEAKER_01"]

    def test_plain_iterable_of_pairs(self):
        pairs = [(_Turn(0.0, 1.0), "A"), (_Turn(0.5, 1.5), "B")]
        segs = _to_segments(_DiarizeOutput(pairs))
        assert [s.speaker for s in segs] == ["A", "B"]

    def test_zero_length_turns_are_dropped(self):
        segs = _to_segments(_Annotation([(_Turn(1.0, 1.0), "_", "A")]))
        assert segs == []

    def test_unexpected_shape_raises_rather_than_guessing(self):
        with pytest.raises(TypeError, match="unexpected diarization item"):
            _to_segments(_DiarizeOutput([(1, 2, 3, 4)]))


class TestExclusiveDiarizationIsNeverUsed:
    """The trap: `exclusive_speaker_diarization` assigns at most ONE speaker per
    instant. Reading it would silently delete every overlap — the overlap mask
    would collapse, `G` would barely run, and `overlap_recall` would read ~0.
    All of which produces a full results table rather than an error.
    """

    def test_overlap_survives(self):
        # The exclusive version resolves the 0.5-1.0 overlap away.
        exclusive = _Annotation([
            (_Turn(0.0, 0.5), "_", "SPEAKER_00"),
            (_Turn(0.5, 1.5), "_", "SPEAKER_01"),
        ])
        segs = _to_segments(_DiarizeOutput(_Annotation(OVERLAPPING), exclusive=exclusive))

        # Two speakers genuinely concurrent between 0.5 and 1.0.
        a = next(s for s in segs if s.speaker == "SPEAKER_00")
        b = next(s for s in segs if s.speaker == "SPEAKER_01")
        assert a.end == pytest.approx(1.0), "overlap was flattened away"
        assert b.start == pytest.approx(0.5)
        assert min(a.end, b.end) - max(a.start, b.start) == pytest.approx(0.5)

    def test_does_not_read_the_exclusive_attribute(self):
        class _Exploding:
            @property
            def exclusive_speaker_diarization(self):
                raise AssertionError("exclusive_speaker_diarization must not be read")

            @property
            def speaker_diarization(self):
                return _Annotation(OVERLAPPING)

        assert len(_to_segments(_Exploding())) == 2
