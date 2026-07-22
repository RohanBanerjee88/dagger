"""Tests for the coarse-to-fine embedding refinement (CLAUDE.md §1, §5 Phase 2).

Two properties matter most: (1) an accepted refinement blends the previous
embedding with a fresh re-embedding of the (purer) extracted estimate, a
rejected one leaves the embedding untouched, and (2) no matter how many rounds
run, the extractor only ever sees the untouched ``x_O`` -- refinement changes
what embedding is fed in, never what audio is fed in.
"""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import FakeSpeakerEncoder, fake_encoder  # noqa: F401,E402

import dagger.refine.coarse_to_fine as coarse_to_fine_module
from dagger.audio.provenance import original_mixture
from dagger.extract.base import Extractor
from dagger.gate.confidence import GateResult
from dagger.reconstruct.stitch import reconstruct_all
from dagger.refine.coarse_to_fine import _longest_run, refine_embeddings

SAMPLE_RATE = 8000


class _AddEmbeddingExtractor(Extractor):
    def _extract(self, x_O, embedding):
        return x_O + float(embedding[0])


class _RecordingExtractor(Extractor):
    def __init__(self):
        self.calls: list[np.ndarray] = []

    def _extract(self, x_O, embedding):
        self.calls.append(x_O.copy())
        return x_O + float(embedding[0])


def _scene():
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
    return x, x_O, activity, solo


class TestLongestRun:
    def test_no_true_returns_none(self):
        assert _longest_run(np.array([0, 0, 0])) is None

    def test_all_true(self):
        assert _longest_run(np.array([1, 1, 1])) == (0, 3)

    def test_picks_the_longest_of_several_runs(self):
        assert _longest_run(np.array([1, 0, 1, 1, 1, 0, 1])) == (2, 5)


class TestRefineEmbeddingsAcceptReject:
    _gate_kwargs = dict(tau_margin=0.0, max_mean_variance=1.0, min_vad_coverage=0.0, max_artifact_score=10.0)

    def _initial_embeddings_and_variances(self):
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variances = np.array([[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]])
        return embeddings, variances

    def test_accepted_update_is_the_blended_mean(self, monkeypatch, fake_encoder):
        x, x_O, activity, solo = _scene()
        extractor = _AddEmbeddingExtractor()
        embeddings, variances = self._initial_embeddings_and_variances()

        monkeypatch.setattr(
            coarse_to_fine_module, "confidence_gate",
            lambda *a, **k: GateResult(True, 1.0, 1.0, 0.0, "accepted"),
        )

        final_embeddings, round_results = refine_embeddings(
            x, x_O, activity, solo, embeddings, variances, extractor, fake_encoder, SAMPLE_RATE,
            rounds=1, **self._gate_kwargs,
        )

        # Reproduce what the function should have computed internally.
        outputs = reconstruct_all(x, x_O, activity, solo, embeddings, extractor)
        for i in range(2):
            run = _longest_run((activity[i] > 0) & (solo[i] <= 0))
            assert run is not None
            clip = outputs[i][run[0]:run[1]]
            raw = fake_encoder.embed(clip, SAMPLE_RATE)
            expected = 0.5 * embeddings[i] + 0.5 * raw
            np.testing.assert_allclose(final_embeddings[i], expected)
            assert round_results[i].accepted is True

    def test_rejected_update_leaves_embedding_unchanged(self, monkeypatch, fake_encoder):
        x, x_O, activity, solo = _scene()
        extractor = _AddEmbeddingExtractor()
        embeddings, variances = self._initial_embeddings_and_variances()

        monkeypatch.setattr(
            coarse_to_fine_module, "confidence_gate",
            lambda *a, **k: GateResult(False, float("nan"), float("nan"), float("nan"), "rejected"),
        )

        final_embeddings, round_results = refine_embeddings(
            x, x_O, activity, solo, embeddings, variances, extractor, fake_encoder, SAMPLE_RATE,
            rounds=2, **self._gate_kwargs,
        )

        np.testing.assert_allclose(final_embeddings, embeddings)
        assert all(r is not None and r.accepted is False for r in round_results)


class TestAudioAlwaysComesFromXOAcrossRounds:
    def test_extractor_only_ever_sees_the_untouched_x_o(self, fake_encoder):
        x, x_O, activity, solo = _scene()
        extractor = _RecordingExtractor()
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variances = np.array([[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]])

        refine_embeddings(
            x, x_O, activity, solo, embeddings, variances, extractor, fake_encoder, SAMPLE_RATE,
            rounds=3, tau_margin=-10.0, max_mean_variance=10.0, min_vad_coverage=0.0,
            max_artifact_score=100.0,
        )

        assert len(extractor.calls) == 3 * 2  # 3 rounds x 2 speakers
        for call in extractor.calls:
            np.testing.assert_array_equal(call, x_O.samples)


class TestModuleIsolation:
    """Coarse-to-fine must stay structurally incapable of the residual
    anti-pattern (CLAUDE.md §1): it should never import the deflation module,
    and -- since it never needs to build a residual at all -- never import
    TrackedSignal/Provenance either."""

    def test_never_imports_deflation_or_provenance_types(self):
        source = inspect.getsource(coarse_to_fine_module)
        tree = ast.parse(source)
        imported_modules = set()
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
                imported_names.update(alias.name for alias in node.names)
        assert not any("dagger.reconstruct.deflation" in mod for mod in imported_modules)
        assert "TrackedSignal" not in imported_names
        assert "Provenance" not in imported_names
