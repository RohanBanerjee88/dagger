"""Phase 0: waveform I/O (dagger.data.audio_io).

``resample``/``read_wav`` are behind the optional ``[data]`` extra
(soundfile + scipy); these tests skip cleanly when it isn't installed rather
than failing the whole suite.
"""

from __future__ import annotations

import numpy as np
import pytest

scipy = pytest.importorskip("scipy")

from dagger.data.audio_io import resample  # noqa: E402


class TestResample:
    def test_same_rate_is_a_noop(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        out = resample(x, 8000, 8000)
        np.testing.assert_array_equal(out, x)

    def test_upsampling_increases_length_by_ratio(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 800))
        out = resample(x, 8000, 16000)
        assert out.shape[0] == 1600

    def test_downsampling_decreases_length_by_ratio(self):
        x = np.sin(np.linspace(0, 4 * np.pi, 1600))
        out = resample(x, 16000, 8000)
        assert out.shape[0] == 800

    def test_preserves_tone_frequency_roughly(self):
        sr_in, sr_out = 8000, 16000
        freq = 200.0
        t = np.arange(sr_in) / sr_in  # 1 second
        x = np.sin(2 * np.pi * freq * t)
        out = resample(x, sr_in, sr_out)
        # Recover dominant frequency via FFT and check it lands near `freq`.
        spectrum = np.abs(np.fft.rfft(out))
        freqs = np.fft.rfftfreq(out.shape[0], d=1.0 / sr_out)
        peak_freq = freqs[np.argmax(spectrum)]
        assert peak_freq == pytest.approx(freq, abs=5.0)
