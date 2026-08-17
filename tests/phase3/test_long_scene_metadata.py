"""The long-scene generator: span selection and the middle-speaker guard.

Both properties encode findings that cost real runs. `_take_span` must return
CONSECUTIVE utterances (a shuffled montage would give the diarizer artificial
acoustic seams to latch onto, flattering the result for the wrong reason), and
the overlap guard must reject 0.5 for 3 speakers, where chain placement closes
the middle speaker's solo window exactly -- the same starvation that produced
the 2-cluster collapse.
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _load():
    spec = importlib.util.spec_from_file_location(
        "build_long_scene_metadata", ROOT / "scripts" / "build_long_scene_metadata.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_long_scene_metadata"] = module
    spec.loader.exec_module(module)
    return module


mod = _load()


def _utterances(n: int, seconds: float = 8.0):
    return [(Path(f"spk/ch/{i:04d}.flac"), seconds) for i in range(n)]


class TestTakeSpan:
    def test_returns_enough_audio(self):
        span = mod._take_span(_utterances(20), 60.0, random.Random(0))
        assert span is not None and len(span) >= 8  # 8 x 8s = 64s >= 60s

    def test_utterances_are_consecutive(self):
        """A shuffled montage would add acoustic jumps a diarizer could exploit."""
        utts = _utterances(20)
        span = mod._take_span(utts, 60.0, random.Random(3))
        order = [int(p.stem) for p in span]
        assert order == list(range(order[0], order[0] + len(order))), order

    def test_none_when_the_speaker_is_too_short(self):
        # 3 x 8s = 24s of audio, 60s requested.
        assert mod._take_span(_utterances(3), 60.0, random.Random(0)) is None

    def test_none_on_an_empty_speaker(self):
        assert mod._take_span([], 10.0, random.Random(0)) is None

    def test_start_varies_across_draws(self):
        """Different scenes drawing the same speaker should not all reuse the
        same opening utterances."""
        utts = _utterances(40)
        starts = {
            mod._take_span(utts, 30.0, random.Random(seed))[0].stem
            for seed in range(12)
        }
        assert len(starts) > 1


class TestMiddleSpeakerGuard:
    """overlap >= 0.5 with 3+ speakers starves the middle speaker."""

    def _run(self, tmp_path, overlap, n_src=3):
        import subprocess

        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_long_scene_metadata.py"),
             "--librispeech-root", str(tmp_path), "--output", str(tmp_path / "o.csv"),
             "--n-src", str(n_src), "--num-scenes", "1",
             "--per-speaker-sec", "50", "--overlap", str(overlap)],
            capture_output=True, text=True,
        )

    def test_overlap_half_is_rejected(self, tmp_path):
        r = self._run(tmp_path, 0.5)
        assert r.returncode != 0
        assert "starves the middle speaker" in (r.stdout + r.stderr)

    def test_overlap_below_half_passes_the_guard(self, tmp_path):
        """It should fail later (empty corpus), not on the overlap check."""
        r = self._run(tmp_path, 0.3)
        combined = r.stdout + r.stderr
        assert "starves the middle speaker" not in combined
        assert "no .flac files found" in combined

    def test_two_speakers_are_exempt(self, tmp_path):
        """With 2 speakers there is no middle speaker to starve."""
        r = self._run(tmp_path, 0.5, n_src=2)
        assert "starves the middle speaker" not in (r.stdout + r.stderr)


class TestPredictedSceneLength:
    def test_prediction_matches_the_documented_formula(self, tmp_path):
        import subprocess

        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "build_long_scene_metadata.py"),
             "--librispeech-root", str(tmp_path), "--output", str(tmp_path / "o.csv"),
             "--n-src", "3", "--num-scenes", "1",
             "--per-speaker-sec", "50", "--overlap", "0.3"],
            capture_output=True, text=True,
        )
        # (1 + (3-1)*(1-0.3)) * 50 = 120 s
        assert "120s = 2.0 min" in r.stdout
