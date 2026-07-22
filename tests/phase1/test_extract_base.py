"""Phase 1: the Extractor ABC and Phase 0's NullExtractor (dagger.extract.base)."""

from __future__ import annotations

import abc

import numpy as np
import pytest

from dagger.audio.provenance import ResidualInAudioPathError, original_mixture
from dagger.extract.base import Extractor, NullExtractor


class TestExtractorAbstractness:
    def test_cannot_instantiate_extractor_directly(self):
        with pytest.raises(TypeError):
            Extractor()

    def test_subclass_must_implement_extract(self):
        class _Incomplete(Extractor):
            pass

        with pytest.raises(TypeError):
            _Incomplete()


class TestNullExtractor:
    def test_returns_silence_matching_input_shape(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0, 4.0]))
        out = NullExtractor().extract(x_O, embedding=np.zeros(8))
        assert out.shape == (4,)
        np.testing.assert_array_equal(out, np.zeros(4))

    def test_ignores_embedding_content(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0]))
        out_a = NullExtractor().extract(x_O, embedding=np.zeros(4))
        out_b = NullExtractor().extract(x_O, embedding=np.ones(4) * 99.0)
        np.testing.assert_array_equal(out_a, out_b)

    def test_rejects_residual_input_via_the_shared_base_class_guard(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0]))
        residual = x_O - np.array([0.1, 0.1, 0.1])
        with pytest.raises(ResidualInAudioPathError):
            NullExtractor().extract(residual, embedding=np.zeros(1))
