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
from dagger.metrics.speaker_similarity import cosine_similarity
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


class TestGateIsNotSelfReferential:
    """The margin must judge the re-embedded clip against the speaker's own
    embedding, never against the blended candidate.

    ``identity_margin`` embeds ``clip`` and compares the result to
    ``embedding_self``. If ``embedding_self`` is ``0.5*e_i + 0.5*raw`` -- half
    made of that very embedding -- the similarity becomes ``cos(theta/2)``
    instead of ``cos(theta)``, which inflates every candidate and inflates the
    worst ones most. That defeats the gate silently: it keeps accepting, and
    its accept rate stops responding to estimate quality at all.
    """

    # min_clip_ms=0.0 disables refine_embeddings' encoder-length floor. `_scene()`
    # is a 5-SAMPLE toy (0.6 ms at 8 kHz) built to exercise blend algebra and gate
    # wiring against a FakeSpeakerEncoder that accepts any length; the floor exists
    # for the real encoder, which cannot embed less than about one mel frame.
    # Without this the overlap runs fall under the floor, every speaker is skipped,
    # and these tests stop testing what they were written to test -- one of them
    # silently, since a skip records accepted=False just as a rejection does.
    _gate_kwargs = dict(
        tau_margin=0.0, max_mean_variance=1.0, min_vad_coverage=0.0,
        max_artifact_score=10.0, min_clip_ms=0.0,
    )

    def test_gate_receives_the_current_embedding_not_the_blend(self, monkeypatch, fake_encoder):
        x, x_O, activity, solo = _scene()
        extractor = _AddEmbeddingExtractor()
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variances = np.array([[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]])
        seen: list[tuple[np.ndarray, np.ndarray]] = []

        def recording_gate(estimate, sample_rate, embedding_self, embeddings_others, *a, **k):
            seen.append((np.array(embedding_self), np.array(k["precomputed_embedding"])))
            return GateResult(False, 1.0, 1.0, 0.0, "rejected")

        monkeypatch.setattr(coarse_to_fine_module, "confidence_gate", recording_gate)

        refine_embeddings(
            x, x_O, activity, solo, embeddings, variances, extractor, fake_encoder, SAMPLE_RATE,
            rounds=1, **self._gate_kwargs,
        )

        assert seen, "gate was never called"
        for speaker_index, (embedding_self, raw) in enumerate(seen):
            np.testing.assert_allclose(embedding_self, embeddings[speaker_index])
            # The blend is what would have been passed before the fix; assert we
            # are not passing it, and that the reference is not contaminated by
            # the clip's own embedding.
            blended = 0.5 * embeddings[speaker_index] + 0.5 * raw
            assert not np.allclose(embedding_self, blended), (
                "gate reference is the blended candidate -- the margin is self-referential"
            )

    def test_a_wrong_candidate_scores_lower_than_the_blend_would_have(self, fake_encoder):
        """The concrete consequence, on the margin arithmetic itself.

        For a candidate at angle theta from the enrollment, an honest reference
        gives cos(theta) while a blended one gives cos(theta/2). At 90 degrees
        that is 0.00 vs 0.71 -- the difference between rejecting a completely
        wrong candidate and comfortably accepting it.
        """
        enrollment = np.array([1.0, 0.0])
        wrong = np.array([0.0, 1.0])  # 90 degrees away -- maximally wrong
        blended = 0.5 * enrollment + 0.5 * wrong

        honest = cosine_similarity(wrong, enrollment)
        self_referential = cosine_similarity(wrong, blended)

        assert honest == pytest.approx(0.0, abs=1e-9)
        assert self_referential == pytest.approx(np.sqrt(0.5), abs=1e-9)
        assert self_referential > honest


class TestRefineEmbeddingsAcceptReject:
    # min_clip_ms=0.0 disables refine_embeddings' encoder-length floor. `_scene()`
    # is a 5-SAMPLE toy (0.6 ms at 8 kHz) built to exercise blend algebra and gate
    # wiring against a FakeSpeakerEncoder that accepts any length; the floor exists
    # for the real encoder, which cannot embed less than about one mel frame.
    # Without this the overlap runs fall under the floor, every speaker is skipped,
    # and these tests stop testing what they were written to test -- one of them
    # silently, since a skip records accepted=False just as a rejection does.
    _gate_kwargs = dict(
        tau_margin=0.0, max_mean_variance=1.0, min_vad_coverage=0.0,
        max_artifact_score=10.0, min_clip_ms=0.0,
    )

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
            assert round_results[0][i].accepted is True
        assert len(round_results) == 1  # one entry per round, not just the last

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
        # Every round is kept, not just the last, so a caller can tell a gate
        # that rejected from round 0 apart from one that only rejected later.
        assert len(round_results) == 2
        assert all(
            r is not None and r.accepted is False
            for per_speaker in round_results
            for r in per_speaker
        )


class TestAudioAlwaysComesFromXOAcrossRounds:
    def test_extractor_only_ever_sees_the_untouched_x_o(self, fake_encoder):
        x, x_O, activity, solo = _scene()
        extractor = _RecordingExtractor()
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variances = np.array([[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]])

        refine_embeddings(
            x, x_O, activity, solo, embeddings, variances, extractor, fake_encoder, SAMPLE_RATE,
            rounds=3, tau_margin=-10.0, max_mean_variance=10.0, min_vad_coverage=0.0,
            max_artifact_score=100.0, min_clip_ms=0.0,  # 5-sample toy scene; see above
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


class TestTheReportedDefaultAndTheCodeDefaultAgree:
    """CLAUDE.md has recorded `refine.rounds: 0` as the default since Phase 2's
    close-out; the code said `2` until 2026-08-25.

    The mismatch moved no committed number -- every config with a `refine:` block
    sets `rounds` explicitly -- but it meant any NEW config omitting the key
    silently switched on a stage now measured at **+0.002 dB** even with a
    perfect candidate and an open gate (Stage B Session 3, Q4b), and bounded at
    **<= +0.18 dB** on the acceptance-rule axis (Stage B Session 1).

    Pinned here because nothing else would catch the drift recurring: the two
    defaults lived in different files from each other and from the prose that
    described them, and the suite stayed green throughout.
    """

    def test_refine_embeddings_defaults_to_no_rounds(self):
        assert inspect.signature(refine_embeddings).parameters["rounds"].default == 0

    @pytest.mark.parametrize("script", ["run_phase2.py", "run_phase3.py"])
    def test_the_entrypoints_default_to_no_rounds(self, script):
        source = (Path(__file__).resolve().parents[2] / "scripts" / script).read_text()
        assert '"rounds", 2' not in source, (
            f"{script} still falls back to 2 rounds when a config omits "
            "`refine.rounds`; the reported default is 0"
        )
        assert '"rounds", 0' in source

    def test_this_does_not_disable_refinement_for_configs_that_ask_for_it(self):
        """The Phase 2 DoD CSVs were produced with `rounds: 2` (every
        `coarse_to_fine` row records it), so reproduction depends on an explicit
        setting still working. This guards the default, not the feature."""
        import dagger.refine.coarse_to_fine as mod
        assert "rounds" in inspect.signature(mod.refine_embeddings).parameters
