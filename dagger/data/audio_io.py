"""Waveform I/O: read a source file to mono float64 at a target sample rate.

Kept behind a thin wrapper so the rest of the data layer never touches
``soundfile``/``scipy`` directly, and so the heavy imports stay lazy (Phase 0's
core is numpy-only; these arrive with the optional ``[data]`` extra).

Naming note: ``sample_rate`` here is always an audio rate in Hz (e.g. 44100 ->
8000). It is unrelated to the speaker symbols (``s_i``); the data layer never
abbreviates sample rate to ``sr`` precisely to avoid that collision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_wav(path: str | Path, target_sample_rate: int) -> np.ndarray:
    """Read ``path`` as a mono float64 waveform resampled to ``target_sample_rate``.

    Multi-channel files are downmixed by averaging channels. Resampling uses a
    polyphase filter (``scipy.signal.resample_poly``), which is band-limited and
    avoids the ringing of naive FFT resampling.
    """
    import soundfile as sf

    samples, file_sample_rate = sf.read(str(path), dtype="float64", always_2d=True)
    samples = samples.mean(axis=1)  # [T, C] -> [T], mono downmix
    return resample(samples, file_sample_rate, target_sample_rate)


def resample(
    samples: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Resample a 1-D waveform from ``source_sample_rate`` to ``target_sample_rate``.

    A no-op when the two rates are equal. Both are audio rates in Hz.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if source_sample_rate == target_sample_rate:
        return samples

    from math import gcd

    from scipy.signal import resample_poly

    common = gcd(int(source_sample_rate), int(target_sample_rate))
    up = int(target_sample_rate) // common
    down = int(source_sample_rate) // common
    return resample_poly(samples, up, down).astype(np.float64)
