"""Tests for the deliberate deflation anti-pattern (CLAUDE.md §1, §5 Phase 2).

This baseline exists ONLY to demonstrate that subtracting a running estimate
and re-extracting from the residual accumulates error -- so these tests check
two things: (1) the module is honest about doing exactly that (the residual it
builds really is tagged ``Provenance.RESIDUAL``, and the guarded
``Extractor.extract()`` path really would refuse it), and (2) the gated
variant successfully prevents a rejected estimate from contaminating the next
speaker's input, while the ungated variant does not.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import dagger.reconstruct.deflation as deflation_module
from dagger.audio.provenance import (
    Provenance,
    ResidualInAudioPathError,
    TrackedSignal,
    original_mixture,
)
from dagger.extract.base import Extractor
from dagger.gate.confidence import GateResult
from dagger.reconstruct.deflation import _extract_from_residual, reconstruct_all_deflation
from dagger.reconstruct.stitch import crossfade_windows


class _EchoExtractor(Extractor):
    def _extract(self, x_O, embedding):
        return x_O * 2.0


class _AddEmbeddingExtractor(Extractor):
    """A deterministic extractor whose output depends on both its input and the
    embedding, so the test can distinguish "extracted from x_O" from "extracted
    from a residual" by exact value."""

    def _extract(self, x_O, embedding):
        return x_O + float(embedding[0])


class TestExtractFromResidualBypassesGuard:
    def test_bypasses_the_guard_that_extract_enforces(self):
        x_O = original_mixture(np.array([1.0, 2.0, 3.0, 4.0]))
        prior_estimate = TrackedSignal(np.array([0.1, 0.1, 0.1, 0.1]), Provenance.DERIVED)
        residual = x_O - prior_estimate
        assert residual.provenance is Provenance.RESIDUAL

        echo = _EchoExtractor()
        with pytest.raises(ResidualInAudioPathError):
            echo.extract(residual, embedding=np.zeros(1))  # the guarded path refuses

        out = _extract_from_residual(echo, residual, embedding=np.zeros(1))  # the deliberate bypass
        np.testing.assert_allclose(out, residual.samples * 2.0)


class TestReconstructAllDeflationAccumulatesWithoutAGate:
    """A 2-speaker synthetic scene, controlled enough to check exact values."""

    def _scene(self):
        x = original_mixture(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), label="x")
        overlap = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
        x_O = x.masked(overlap, label="x_O")
        activity = np.array(
            [
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0],
            ]
        )
        solo = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
        embeddings = np.array([[10.0], [20.0]])
        return x, x_O, activity, solo, embeddings

    def test_ungated_second_speaker_extracts_from_a_residual(self):
        x, x_O, activity, solo, embeddings = self._scene()
        extractor = _AddEmbeddingExtractor()
        outputs, gate_results = reconstruct_all_deflation(
            x, x_O, activity, solo, embeddings, extractor, order=[0, 1], gate_fn=None
        )
        assert gate_results == [None, None]

        # Speaker 0 extracts straight from x_O (nothing prior to subtract yet).
        g_out_0 = x_O.samples + 10.0
        _, w_O0 = crossfade_windows(solo[0], activity[0])
        _, w_O1 = crossfade_windows(solo[1], activity[1])
        w_E1, _ = crossfade_windows(solo[1], activity[1])

        # Speaker 1 extracts from x_O minus speaker 0's accepted contribution.
        residual_for_1 = x_O.samples - g_out_0 * w_O0
        g_out_1 = residual_for_1 + 20.0
        expected_1 = np.asarray(x, dtype=np.float64) * w_E1 + g_out_1 * w_O1
        np.testing.assert_allclose(outputs[1], expected_1)

        # And this is NOT the same as extracting speaker 1 fresh from x_O.
        fresh_g_out_1 = x_O.samples + 20.0
        fresh_expected_1 = np.asarray(x, dtype=np.float64) * w_E1 + fresh_g_out_1 * w_O1
        assert not np.allclose(outputs[1], fresh_expected_1)

    def test_gated_reject_all_never_contaminates_the_next_speaker(self):
        x, x_O, activity, solo, embeddings = self._scene()
        extractor = _AddEmbeddingExtractor()

        def reject_everything(speaker_idx, estimate):
            return GateResult(False, float("nan"), float("nan"), float("nan"), "rejected")

        outputs, gate_results = reconstruct_all_deflation(
            x, x_O, activity, solo, embeddings, extractor, order=[0, 1], gate_fn=reject_everything
        )
        assert all(r is not None and r.accepted is False for r in gate_results)

        # Because speaker 0's estimate was rejected, speaker 1 still extracts
        # from the untouched x_O -- identical to a from-scratch (non-deflation)
        # extraction.
        w_E1, w_O1 = crossfade_windows(solo[1], activity[1])
        fresh_g_out_1 = x_O.samples + 20.0
        expected_1 = np.asarray(x, dtype=np.float64) * w_E1 + fresh_g_out_1 * w_O1
        np.testing.assert_allclose(outputs[1], expected_1)

    def test_gated_accept_all_matches_ungated(self):
        x, x_O, activity, solo, embeddings = self._scene()
        extractor = _AddEmbeddingExtractor()

        def accept_everything(speaker_idx, estimate):
            return GateResult(True, 1.0, 1.0, 0.0, "accepted")

        ungated_outputs, _ = reconstruct_all_deflation(
            x, x_O, activity, solo, embeddings, extractor, order=[0, 1], gate_fn=None
        )
        gated_outputs, gate_results = reconstruct_all_deflation(
            x, x_O, activity, solo, embeddings, extractor, order=[0, 1], gate_fn=accept_everything
        )
        assert all(r is not None and r.accepted for r in gate_results)
        np.testing.assert_allclose(ungated_outputs, gated_outputs)


class TestModuleIsolation:
    """The deflation anti-pattern must stay structurally quarantined from the
    accumulation-free code path (CLAUDE.md §1)."""

    def test_deflation_module_never_imports_refine(self):
        source = inspect.getsource(deflation_module)
        tree = ast.parse(source)
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        assert not any("dagger.refine" in mod for mod in imported_modules)
