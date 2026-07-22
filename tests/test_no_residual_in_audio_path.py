"""CLAUDE.md guardrail (§1, §6.7): no output tensor may come from a residual.

This is the single most important test in the repo. Every speaker's output
waveform must be extracted from the untouched overlap mixture ``x_O`` --
never from ``x_O`` minus a running estimate. If this file starts failing,
something is deflating a residual into the extractor; stop and re-read
CLAUDE.md §1 before touching anything else.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.audio.provenance import (
    Provenance,
    ResidualInAudioPathError,
    TrackedSignal,
    original_mixture,
    require_original_mixture,
)
from dagger.extract.base import Extractor, NullExtractor
from dagger.reconstruct.stitch import reconstruct_all, reconstruct_speaker


class TestTrackedSignalProvenance:
    def test_original_mixture_helper_tags_correctly(self):
        sig = original_mixture(np.array([1.0, 2.0, 3.0]), label="x")
        assert sig.provenance is Provenance.ORIGINAL_MIXTURE
        assert sig.is_original_mixture
        assert sig.label == "x"

    def test_masking_preserves_original_mixture_provenance(self):
        x = original_mixture(np.array([1.0, 2.0, 3.0, 4.0]), label="x")
        x_O = x.masked(np.array([0.0, 1.0, 1.0, 0.0]), label="x_O")
        assert x_O.is_original_mixture
        np.testing.assert_array_equal(x_O.samples, [0.0, 2.0, 3.0, 0.0])

    def test_subtraction_produces_residual(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0]), label="x_O")
        s_hat = TrackedSignal(np.array([0.5, 0.5, 0.5]), Provenance.DERIVED, label="s_hat_0")
        residual = x_O - s_hat
        assert residual.provenance is Provenance.RESIDUAL
        assert not residual.is_original_mixture
        np.testing.assert_allclose(residual.samples, [0.5, 1.5, 2.5])

    def test_subtracting_raw_array_also_produces_residual(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0]))
        residual = x_O - np.array([1.0, 1.0, 1.0])
        assert residual.provenance is Provenance.RESIDUAL

    def test_deflation_chain_stays_residual_even_after_further_masking(self):
        # x_O -> subtract estimate -> mask again (e.g. restrict to a region):
        # still a residual. Masking never launders a residual back into an
        # original mixture.
        x_O = original_mixture(np.array([1.0, 2.0, 3.0, 4.0]))
        s_hat = TrackedSignal(np.array([0.1, 0.1, 0.1, 0.1]), Provenance.DERIVED)
        residual = x_O - s_hat
        masked_residual = residual.masked(np.array([1.0, 1.0, 0.0, 0.0]))
        assert masked_residual.provenance is Provenance.RESIDUAL


class TestRequireOriginalMixture:
    def test_passes_through_original_mixture_samples(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0]))
        out = require_original_mixture(x_O)
        np.testing.assert_array_equal(out, [1.0, 2.0, 3.0])

    def test_raises_on_residual(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0]))
        s_hat = TrackedSignal(np.array([0.5, 0.5, 0.5]), Provenance.DERIVED)
        residual = x_O - s_hat
        with pytest.raises(ResidualInAudioPathError):
            require_original_mixture(residual)

    def test_raises_on_derived(self):
        derived = TrackedSignal(np.array([1.0, 2.0]), Provenance.DERIVED)
        with pytest.raises(ResidualInAudioPathError):
            require_original_mixture(derived)

    def test_raises_type_error_on_raw_ndarray(self):
        with pytest.raises(TypeError):
            require_original_mixture(np.array([1.0, 2.0, 3.0]))

    def test_error_message_names_the_context(self):
        residual = original_mixture(np.array([1.0])) - np.array([1.0])
        with pytest.raises(ResidualInAudioPathError, match="my_extractor"):
            require_original_mixture(residual, context="my_extractor")


class TestExtractorGuardsResidual:
    """The ABC-level guard: no Extractor subclass can bypass it."""

    def test_null_extractor_accepts_original_mixture(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0, 4.0]))
        out = NullExtractor().extract(x_O, embedding=np.zeros(4))
        np.testing.assert_array_equal(out, np.zeros(4))

    def test_deflation_anti_pattern_trips_the_guard(self):
        """The exact anti-pattern CLAUDE.md §1 forbids: subtract, then extract."""
        x_O = original_mixture(np.array([1.0, 2.0, 3.0, 4.0]))
        s_hat_prev = TrackedSignal(np.array([0.1, 0.1, 0.1, 0.1]), Provenance.DERIVED)
        deflated_residual = x_O - s_hat_prev  # the forbidden running residual
        with pytest.raises(ResidualInAudioPathError):
            NullExtractor().extract(deflated_residual, embedding=np.zeros(4))

    def test_guard_applies_to_any_extractor_subclass(self):
        """A hypothetical extractor with a real _extract implementation still
        can't be handed a residual -- the base class checks before dispatch."""

        class _EchoExtractor(Extractor):
            def _extract(self, x_O, embedding):
                return x_O * 2.0

        echo = _EchoExtractor()
        x_O = original_mixture(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(echo.extract(x_O, np.zeros(1)), [2.0, 4.0])

        residual = x_O - np.array([0.5, 0.5])
        with pytest.raises(ResidualInAudioPathError):
            echo.extract(residual, np.zeros(1))


class _SpyExtractor(Extractor):
    """Records the provenance of every ``x_O`` it is actually called with."""

    def __init__(self):
        self.seen_provenance: list[Provenance] = []

    def _extract(self, x_O, embedding):
        # By the time _extract runs, require_original_mixture already passed,
        # so provenance is implicitly ORIGINAL_MIXTURE -- record it explicitly
        # anyway so the assertion below is about behavior, not trust.
        self.seen_provenance.append(Provenance.ORIGINAL_MIXTURE)
        return np.zeros_like(x_O)


class TestReconstructNeverFeedsResidual:
    """The reconstruction stitcher (Phase 0/1's only path to output audio)
    must call G on the same untouched x_O for every speaker -- never on a
    residual built from another speaker's output (that would be the
    accumulation bug the whole project exists to avoid)."""

    def test_reconstruct_speaker_feeds_original_mixture(self):
        spy = _SpyExtractor()
        x = original_mixture(np.array([1.0, 2.0, 3.0, 4.0]), label="x")
        x_O = x.masked(np.array([0.0, 1.0, 1.0, 0.0]), label="x_O")
        activity_i = np.array([1.0, 1.0, 1.0, 1.0])
        solo_i = np.array([1.0, 0.0, 0.0, 1.0])
        reconstruct_speaker(x, x_O, activity_i, solo_i, embedding=np.zeros(2), extractor=spy)
        assert spy.seen_provenance == [Provenance.ORIGINAL_MIXTURE]

    def test_reconstruct_all_feeds_the_same_x_o_to_every_speaker(self):
        """Accumulation-free property: speaker i+1's extraction must not depend
        on speaker i's output, so every call receives byte-identical x_O."""

        class _RecordingExtractor(Extractor):
            def __init__(self):
                self.calls: list[np.ndarray] = []

            def _extract(self, x_O, embedding):
                self.calls.append(x_O.copy())
                return np.zeros_like(x_O)

        extractor = _RecordingExtractor()
        x = original_mixture(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), label="x")
        overlap = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        x_O = x.masked(overlap, label="x_O")
        activity = np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        )
        solo = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        reconstruct_all(x, x_O, activity, solo, embeddings=None, extractor=extractor)

        assert len(extractor.calls) == 3  # one call per speaker
        for call in extractor.calls[1:]:
            np.testing.assert_array_equal(call, extractor.calls[0])
            np.testing.assert_array_equal(call, x_O.samples)
