"""Guardrail §6.7: fail if any output audio was produced from a residual.

This is the cheap insurance against the one mistake that would silently
invalidate the paper (CLAUDE.md §1): extracting a speaker from ``x_O`` minus a
running estimate instead of from the untouched ``x_O``.
"""

import numpy as np
import pytest

from dagger.audio.provenance import (
    Provenance,
    ResidualInAudioPathError,
    TrackedSignal,
    original_mixture,
)
from dagger.extract.base import NullExtractor


def test_masked_original_mixture_stays_original():
    """x_O is a masked slice of the mixture -> still a legal extractor input."""
    x = original_mixture(np.ones(16), label="x")
    overlap = np.array([0, 0, 1, 1, 1, 1, 0, 0] * 2, dtype=float)
    x_O = x.masked(overlap, label="x_O")
    assert x_O.provenance is Provenance.ORIGINAL_MIXTURE
    # The correct pipeline: extracting from x_O must NOT raise.
    NullExtractor().extract(x_O, embedding=np.zeros(4))


def test_subtraction_marks_signal_as_residual():
    x_O = original_mixture(np.arange(8, dtype=float), label="x_O")
    s_hat = TrackedSignal(np.ones(8), provenance=Provenance.DERIVED, label="s_hat_0")
    residual = x_O - s_hat
    assert residual.provenance is Provenance.RESIDUAL


def test_extractor_rejects_residual_input():
    """The deflation anti-pattern: feed x_O - s_hat into G -> must raise."""
    x_O = original_mixture(np.arange(8, dtype=float), label="x_O")
    s_hat = TrackedSignal(np.ones(8), provenance=Provenance.DERIVED, label="s_hat_0")
    residual = x_O - s_hat  # x_O minus a running estimate == the forbidden residual

    with pytest.raises(ResidualInAudioPathError):
        NullExtractor().extract(residual, embedding=np.zeros(4))


def test_extractor_rejects_derived_input():
    derived = TrackedSignal(np.ones(8), provenance=Provenance.DERIVED, label="g_out")
    with pytest.raises(ResidualInAudioPathError):
        NullExtractor().extract(derived, embedding=np.zeros(4))


def test_extractor_rejects_untracked_array():
    """Passing a bare ndarray bypasses provenance checking -> reject it."""
    with pytest.raises(TypeError):
        NullExtractor().extract(np.ones(8), embedding=np.zeros(4))
