"""Phase 1: eval-only speaker similarity / margin (dagger.metrics.speaker_similarity).

CLAUDE.md's "eval encoder != training encoder" guardrail is non-negotiable,
so besides the numeric behavior of the margin, this file asserts the
metric-hygiene property architecturally: this module must never import
``dagger.enroll.encoder`` (the training-side phi), so the eval encoder
physically cannot alias phi.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import dagger.metrics.speaker_similarity as speaker_similarity_module
from dagger.metrics.speaker_similarity import cosine_similarity, eval_enroll_and_margin

SAMPLE_RATE = 1000


class TestCosineSimilarity:
    def test_identical_vectors_give_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_opposite_vectors_give_minus_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, -v) == pytest.approx(-1.0)

    def test_orthogonal_vectors_give_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_norm_vector_gives_nan(self):
        a = np.zeros(3)
        b = np.array([1.0, 2.0, 3.0])
        assert np.isnan(cosine_similarity(a, b))

    def test_scale_invariant(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        assert cosine_similarity(a, 10.0 * b) == pytest.approx(cosine_similarity(a, b))


class TestMetricHygiene:
    def test_never_imports_the_training_encoder_module(self):
        """Architectural guard: dagger.metrics.speaker_similarity must not
        import dagger.enroll.encoder (phi), so the eval encoder can never
        accidentally alias the training encoder."""
        source = inspect.getsource(speaker_similarity_module)
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        assert not any("enroll.encoder" in mod for mod in imported_modules)

    def test_titanet_encoder_symbol_is_not_present(self):
        assert not hasattr(speaker_similarity_module, "TitaNetEncoder")


class _FakeEvalEncoder:
    """Duck-typed stand-in for EvalSpeakerEncoder: deterministic, dependency-free."""

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        x = np.asarray(waveform, dtype=np.float64)
        return np.array([x.mean(), float(np.sqrt(np.mean(x ** 2)))])


class TestEvalEnrollAndMargin:
    def test_perfectly_reconstructed_speakers_get_positive_margins(self):
        n = 200
        mixture = np.zeros(n)
        activity = np.zeros((2, n))
        solo = np.zeros((2, n))
        outputs = np.zeros((2, n))

        # Speaker 0: solo + reconstructed as a +1 DC tone; speaker 1: -1 DC tone.
        activity[0, :100] = 1.0
        solo[0, :100] = 1.0
        mixture[:100] = 1.0
        outputs[0, :100] = 1.0

        activity[1, 100:] = 1.0
        solo[1, 100:] = 1.0
        mixture[100:] = -1.0
        outputs[1, 100:] = -1.0

        margins = eval_enroll_and_margin(
            mixture, solo, activity, outputs, SAMPLE_RATE, _FakeEvalEncoder(),
            k=1, min_clip_ms=0.0,
        )
        assert len(margins) == 2
        assert all(m > 0 for m in margins)

    def test_swapped_outputs_get_negative_margins(self):
        """If speaker 0's output actually sounds like speaker 1, the margin
        must go negative -- this is the whole point of the diagnostic."""
        n = 200
        mixture = np.zeros(n)
        activity = np.zeros((2, n))
        solo = np.zeros((2, n))
        outputs = np.zeros((2, n))

        activity[0, :100] = 1.0
        solo[0, :100] = 1.0
        mixture[:100] = 1.0

        activity[1, 100:] = 1.0
        solo[1, 100:] = 1.0
        mixture[100:] = -1.0

        # Deliberately swapped: output[0] sounds like speaker 1 and vice versa.
        outputs[0, :100] = -1.0
        outputs[1, 100:] = 1.0

        margins = eval_enroll_and_margin(
            mixture, solo, activity, outputs, SAMPLE_RATE, _FakeEvalEncoder(),
            k=1, min_clip_ms=0.0,
        )
        assert all(m < 0 for m in margins)
