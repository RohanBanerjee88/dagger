"""The oracle-refinement ceiling (Phase 3 Stage B item 2).

Refinement is net-negative in every regime measured so far, but "never
positive" is not provable by accumulating negatives -- and the mechanism
explains the pattern without settling it, since a bad candidate could be the
extractor's fault rather than refinement's. Substituting an acceptance rule that
can see the ground truth separates the two.

Two things get pinned here, and the first is the more important:

1. **``accept_fn=None`` changes nothing.** The ceiling is a scoring-time
   instrument bolted onto the deployable path, so the deployable path must be
   provably untouched by its presence.
2. The oracle rule accepts genuine improvements, rejects degradations, and maps
   its row indices onto the right sources -- a wrong ``kept`` map would score
   each speaker against another speaker's audio and report a confidently wrong
   ceiling, with nothing failing.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.audio.provenance import original_mixture  # noqa: E402
from dagger.extract.base import Extractor  # noqa: E402
from dagger.refine.coarse_to_fine import (  # noqa: E402
    CEILING_ACCEPT_GATE_REJECT,
    CEILING_REJECT_GATE_ACCEPT,
    refine_embeddings,
)
from dagger.refine.oracle_ceiling import make_oracle_accept_fn  # noqa: E402

SAMPLE_RATE = 8000

_GATE_KWARGS = dict(
    tau_margin=0.0, max_mean_variance=1.0, min_vad_coverage=0.0,
    max_artifact_score=10.0, min_clip_ms=0.0,
)


class _AddEmbeddingExtractor(Extractor):
    def _extract(self, x_O, embedding):
        return x_O + float(embedding[0])


def _scene():
    """The same 5-sample toy the Phase 2 refinement tests use."""
    x = original_mixture(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), label="x")
    overlap = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    x_O = x.masked(overlap, label="x_O")
    activity = np.array([
        [1.0, 1.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0, 1.0, 1.0],
    ])
    solo = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    return x, x_O, activity, solo


def _run(accept_fn, fake_encoder, rounds=2):
    x, x_O, activity, solo = _scene()
    return refine_embeddings(
        x, x_O, activity, solo,
        np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        np.array([[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]]),
        _AddEmbeddingExtractor(), fake_encoder, SAMPLE_RATE,
        rounds=rounds, accept_fn=accept_fn, **_GATE_KWARGS,
    )


class TestDefaultPathIsUntouched:
    def test_none_matches_a_gate_mirroring_accept_fn(self, fake_encoder):
        """An ``accept_fn`` that reproduces the gate must reproduce the result.

        This is the equivalence that makes the feature safe: the only thing
        ``accept_fn`` changes is WHICH predicate decides, so a predicate equal
        to the gate's own verdict has to land on identical embeddings.
        """
        gate_verdicts: list[bool] = []
        baseline_embeddings, baseline_rounds = _run(None, fake_encoder)
        for per_speaker in baseline_rounds:
            for result in per_speaker:
                if result is not None:
                    gate_verdicts.append(result.accepted)

        replay = iter(gate_verdicts)
        mirrored, mirrored_rounds = _run(
            lambda i, cand, cur: next(replay), fake_encoder
        )

        np.testing.assert_allclose(mirrored, baseline_embeddings)
        assert [
            r.accepted for per in mirrored_rounds for r in per if r is not None
        ] == gate_verdicts

    def test_accept_fn_none_needs_no_ground_truth(self, fake_encoder):
        """Sanity: the default path never touches sources, so it still runs."""
        embeddings, rounds = _run(None, fake_encoder)
        assert embeddings.shape == (2, 3)
        assert len(rounds) == 2


class TestOracleRuleJudgesOutcomes:
    def test_accepts_a_candidate_that_improves_si_sdr(self):
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        accept = make_oracle_accept_fn([target])

        better = target * 0.99          # nearly the target
        worse = target + 5.0            # badly offset
        assert accept(0, better, worse) is True

    def test_rejects_a_candidate_that_degrades_si_sdr(self):
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        accept = make_oracle_accept_fn([target])

        assert accept(0, target + 5.0, target * 0.99) is False

    def test_ties_reject(self):
        """A no-op must not count as a win.

        With a strict inequality the measured ceiling is a LOWER bound on the
        achievable one -- the safe direction, since the negative reading is the
        publishable outcome.
        """
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        accept = make_oracle_accept_fn([target])
        estimate = target * 0.5
        assert accept(0, estimate, estimate) is False

    def test_undefined_comparisons_reject(self):
        """A silent target makes "better" undefined; that is not an improvement."""
        accept = make_oracle_accept_fn([np.zeros(5)])
        assert accept(0, np.ones(5), np.zeros(5)) is False

    def test_a_spurious_cluster_rejects(self):
        """A row with no ground-truth counterpart has nothing to get closer to."""
        accept = make_oracle_accept_fn([None])
        assert accept(0, np.ones(5), np.zeros(5)) is False

    def test_rows_are_indexed_by_refiner_row_not_source(self):
        """``score_scene`` drops unenrollable clusters and re-indexes.

        Refiner row 0 is then whichever source survived, not source 0. Taking
        already-restricted per-row sources is what removes that mapping from the
        caller -- getting it wrong would score each speaker against another
        speaker's audio with nothing failing.
        """
        speaker_a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        speaker_b = np.array([3.0, 0.0, -3.0, 0.0, 3.0])

        # Cluster for speaker_a was dropped: the only refiner row is speaker_b.
        accept = make_oracle_accept_fn([speaker_b])
        assert accept(0, speaker_b * 0.99, speaker_b + 9.0) is True
        # Judged against the other speaker, the same pair goes the other way.
        assert make_oracle_accept_fn([speaker_a])(
            0, speaker_b * 0.99, speaker_b + 9.0
        ) is False


class TestDisagreementIsRecorded:
    def test_override_labels_where_the_gate_would_have_differed(self, fake_encoder):
        """The 2x2 lands in ``reason``, so no CSV schema has to change.

        ``ceiling_accept_gate_would_reject`` is the headline quantity of a
        ceiling run: headroom the deployable gate cannot reach.
        """
        _, baseline_rounds = _run(None, fake_encoder)
        gate_said = [
            r.accepted for per in baseline_rounds for r in per if r is not None
        ]
        assert gate_said, "fixture produced no gate decisions"

        # Invert every gate verdict, so every decision is a disagreement.
        replay = iter([not v for v in gate_said])
        _, rounds = _run(lambda i, cand, cur: next(replay), fake_encoder)

        reasons = [r.reason for per in rounds for r in per if r is not None]
        assert all(
            reason in (CEILING_ACCEPT_GATE_REJECT, CEILING_REJECT_GATE_ACCEPT)
            for reason in reasons
        ), reasons

    def test_accepted_reports_what_was_committed(self, fake_encoder):
        """``accepted`` must track the audio, not the gate's opinion of it."""
        _, rounds = _run(lambda i, cand, cur: True, fake_encoder)
        decisions = [r for per in rounds for r in per if r is not None]
        assert decisions and all(r.accepted for r in decisions)
