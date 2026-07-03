"""Oracle diarization: RTTM parsing, activity matrix, solo/overlap regions."""

import numpy as np

from dagger.diarize.oracle import (
    activity_matrix,
    overlap_mixture,
    read_rttm,
    solo_overlap_regions,
)
from dagger.audio.provenance import Provenance, original_mixture

RTTM = """\
SPEAKER file 1 0.00 1.00 <NA> <NA> spkA <NA> <NA>
SPEAKER file 1 0.50 1.00 <NA> <NA> spkB <NA> <NA>
"""


def _write(tmp_path, text):
    p = tmp_path / "file.rttm"
    p.write_text(text)
    return str(p)


def test_read_rttm(tmp_path):
    segs = read_rttm(_write(tmp_path, RTTM))
    assert [s.speaker for s in segs] == ["spkA", "spkB"]
    assert segs[0].start == 0.0 and segs[0].end == 1.0
    assert segs[1].start == 0.5 and segs[1].end == 1.5


def test_activity_and_regions():
    sr = 10  # 10 samples/sec for an easy hand-check
    # spkA active [0.0, 1.0) -> samples 0..9; spkB active [0.5, 1.5) -> samples 5..14
    from dagger.diarize.oracle import Segment

    segments = [Segment("spkA", 0.0, 1.0), Segment("spkB", 0.5, 1.0)]
    activity, speakers = activity_matrix(segments, num_samples=15, sample_rate=sr)
    assert speakers == ["spkA", "spkB"]
    assert activity.shape == (2, 15)

    solo, overlap = solo_overlap_regions(activity)
    # samples 0..4: only A (solo A); 5..9: both (overlap); 10..14: only B (solo B)
    assert solo[0, :5].all() and not solo[0, 5:].any()
    assert not solo[1, :10].any() and solo[1, 10:].all()
    assert not overlap[:5].any() and overlap[5:10].all() and not overlap[10:].any()

    # partition: a solo frame is never an overlap frame
    assert not (solo.sum(axis=0).astype(bool) & overlap.astype(bool)).any()


def test_overlap_mixture_is_original_mixture():
    x = original_mixture(np.arange(15, dtype=float))
    overlap = np.zeros(15)
    overlap[5:10] = 1.0
    x_O = overlap_mixture(x, overlap)
    assert x_O.provenance is Provenance.ORIGINAL_MIXTURE
    assert not np.asarray(x_O)[:5].any()
    assert np.asarray(x_O)[5:10].tolist() == list(range(5, 10))
