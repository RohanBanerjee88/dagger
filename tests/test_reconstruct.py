"""Reconstruction: partition of unity + exact solo recovery (Phase 0 DoD)."""

import numpy as np

from dagger.audio.provenance import original_mixture
from dagger.diarize.oracle import (
    Segment,
    activity_matrix,
    overlap_mixture,
    solo_overlap_regions,
)
from dagger.extract.base import NullExtractor
from dagger.reconstruct.stitch import crossfade_windows, reconstruct_all


def test_partition_of_unity_hard():
    solo_i = np.array([1, 1, 0, 0, 0], dtype=float)
    activity_i = np.array([1, 1, 1, 1, 0], dtype=float)
    w_E, w_O = crossfade_windows(solo_i, activity_i, fade=0)
    np.testing.assert_allclose(w_E + w_O, activity_i)
    np.testing.assert_allclose(w_E, solo_i)


def test_partition_of_unity_crossfade():
    # long solo run, then overlap run, so the seam is interior
    solo_i = np.concatenate([np.ones(20), np.zeros(20)])
    activity_i = np.ones(40)
    w_E, w_O = crossfade_windows(solo_i, activity_i, fade=3)
    # partition of unity holds sample-by-sample regardless of the crossfade
    np.testing.assert_allclose(w_E + w_O, activity_i)
    assert np.all(w_E >= -1e-9) and np.all(w_O >= -1e-9)
    # deep in the solo interior w_E == 1; deep in the overlap interior w_E == 0
    assert w_E[5] == 1.0
    assert w_E[34] == 0.0
    # a genuine ramp exists across the seam
    assert 0.0 < w_E[20] < 1.0


def test_exact_solo_recovery():
    """Phase 0 definition of done: copying solo regions recovers solo audio exactly."""
    sr = 100
    # Two distinct sources; mixture = sum. spkA solo [0,0.4), overlap [0.4,0.6), etc.
    n = 100
    t = np.arange(n) / sr
    s0 = np.sin(2 * np.pi * 5 * t)
    s1 = np.sin(2 * np.pi * 11 * t)

    segments = [Segment("A", 0.0, 0.6), Segment("B", 0.4, 0.6)]
    activity, speakers = activity_matrix(segments, num_samples=n, sample_rate=sr)
    # zero each source outside its own activity so the mixture is physically consistent
    s0 = s0 * activity[0]
    s1 = s1 * activity[1]
    mix = s0 + s1

    solo, overlap = solo_overlap_regions(activity)
    x = original_mixture(mix, label="x")
    x_O = overlap_mixture(x, overlap)

    outputs = reconstruct_all(
        x=x, x_O=x_O, activity=activity, solo=solo,
        embeddings=None, extractor=NullExtractor(), fade=0,
    )

    # In solo frames, the reconstruction must equal the clean source exactly.
    solo0 = solo[0].astype(bool)
    solo1 = solo[1].astype(bool)
    np.testing.assert_allclose(outputs[0][solo0], s0[solo0], atol=1e-12)
    np.testing.assert_allclose(outputs[1][solo1], s1[solo1], atol=1e-12)
