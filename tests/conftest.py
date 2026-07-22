"""Shared synthetic-scene helpers for the dagger test suite.

Every fixture here builds small, fast, numpy-only scenes via
``dagger.data.mixing`` + ``dagger.data.activity`` (exact placement windows, no
VAD estimation) so tests never need real corpus audio or the ``[data]``/
``[ml]`` extras unless a specific test opts into them.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.data.activity import segments_from_placement
from dagger.data.mixing import stagger_offsets
from dagger.diarize.oracle import activity_matrix, solo_overlap_regions

SAMPLE_RATE = 8000


def make_tone(length: int, freq_hz: float, sample_rate: int = SAMPLE_RATE, amp: float = 0.5) -> np.ndarray:
    """A deterministic sine tone, used as a stand-in "clean source" clip."""
    t = np.arange(length, dtype=np.float64) / sample_rate
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def build_staggered_scene(
    lengths: list[int],
    overlap: float,
    min_solo: int = 0,
    sample_rate: int = SAMPLE_RATE,
    freqs: list[float] | None = None,
):
    """Build a small staggered multi-speaker scene end to end.

    Returns a dict with ``mixture`` [T], ``sources`` [S, T] (aligned, gain=1,
    zero-padded to common length), ``speakers``, ``offsets``, ``activity``
    [S, T], ``solo`` [S, T], ``overlap`` [T]. Mirrors exactly what
    ``dagger.data.librimix``/``wsj0mix`` produce, minus file I/O.
    """
    num = len(lengths)
    speakers = [f"s{i}" for i in range(num)]
    freqs = freqs or [220.0 * (i + 1) for i in range(num)]
    offsets = stagger_offsets(lengths, overlap=overlap, min_solo=min_solo)

    out_len = max(offsets[i] + lengths[i] for i in range(num))
    sources = np.zeros((num, out_len), dtype=np.float64)
    for i in range(num):
        clip = make_tone(lengths[i], freqs[i], sample_rate=sample_rate)
        sources[i, offsets[i]:offsets[i] + lengths[i]] = clip
    mixture = sources.sum(axis=0)

    segments = segments_from_placement(offsets, lengths, speakers, sample_rate)
    activity, speakers = activity_matrix(segments, num_samples=out_len, sample_rate=sample_rate, speakers=speakers)
    solo, overlap_mask = solo_overlap_regions(activity)

    return {
        "mixture": mixture,
        "sources": sources,
        "speakers": speakers,
        "offsets": offsets,
        "lengths": lengths,
        "sample_rate": sample_rate,
        "activity": activity,
        "solo": solo,
        "overlap": overlap_mask,
    }


@pytest.fixture
def two_speaker_scene():
    """A 2-speaker scene with both a solo lead-in/tail and an overlap middle."""
    return build_staggered_scene(lengths=[4000, 3000], overlap=0.5)


@pytest.fixture
def three_speaker_scene():
    """A 3-speaker scene using ``min_solo`` so every speaker keeps solo time."""
    return build_staggered_scene(lengths=[4000, 2000, 3000], overlap=0.5, min_solo=800)


class FakeSpeakerEncoder:
    """A deterministic, dependency-free stand-in for ``SpeakerEncoder``.

    Embeds a clip as ``[mean, rms, zero-crossing-rate]`` -- cheap, fully
    deterministic, and distinct enough between differently-pitched tones that
    enrollment / margin tests can assert meaningful (non-degenerate) behavior
    without loading TitaNet or WavLM.
    """

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        x = np.asarray(waveform, dtype=np.float64)
        if x.size == 0:
            return np.zeros(3, dtype=np.float64)
        mean = float(x.mean())
        rms = float(np.sqrt(np.mean(x ** 2)))
        signs = np.sign(x)
        signs[signs == 0] = 1.0
        zcr = float(np.mean(signs[:-1] != signs[1:])) if x.size > 1 else 0.0
        return np.array([mean, rms, zcr], dtype=np.float64)


@pytest.fixture
def fake_encoder():
    return FakeSpeakerEncoder()
