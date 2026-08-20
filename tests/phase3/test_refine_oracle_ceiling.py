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


ALL_OVERLAP = np.full(5, 2)          # every sample at depth 2, i.e. fully scored


class TestOracleRuleJudgesOutcomes:
    def test_accepts_a_candidate_that_improves_si_sdr(self):
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        accept = make_oracle_accept_fn([target], ALL_OVERLAP)

        better = target * 0.99          # nearly the target
        worse = target + 5.0            # badly offset
        assert accept(0, better, worse) is True

    def test_rejects_a_candidate_that_degrades_si_sdr(self):
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        accept = make_oracle_accept_fn([target], ALL_OVERLAP)

        assert accept(0, target + 5.0, target * 0.99) is False

    def test_ties_reject(self):
        """A no-op must not count as a win.

        With a strict inequality the measured ceiling is a LOWER bound on the
        achievable one -- the safe direction, since the negative reading is the
        publishable outcome.
        """
        target = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
        accept = make_oracle_accept_fn([target], ALL_OVERLAP)
        estimate = target * 0.5
        assert accept(0, estimate, estimate) is False

    def test_undefined_comparisons_reject(self):
        """A silent target makes "better" undefined; that is not an improvement."""
        accept = make_oracle_accept_fn([np.zeros(5)], ALL_OVERLAP)
        assert accept(0, np.ones(5), np.zeros(5)) is False

    def test_a_spurious_cluster_rejects(self):
        """A row with no ground-truth counterpart has nothing to get closer to."""
        accept = make_oracle_accept_fn([None], ALL_OVERLAP)
        assert accept(0, np.ones(5), np.zeros(5)) is False

    def test_a_scene_with_no_overlap_rejects(self):
        """An empty mask makes the comparison undefined, not favourable."""
        accept = make_oracle_accept_fn([np.ones(5)], np.ones(5))   # all depth 1
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
        accept = make_oracle_accept_fn([speaker_b], ALL_OVERLAP)
        assert accept(0, speaker_b * 0.99, speaker_b + 9.0) is True
        # Judged against the other speaker, the same pair goes the other way.
        assert make_oracle_accept_fn([speaker_a], ALL_OVERLAP)(
            0, speaker_b * 0.99, speaker_b + 9.0
        ) is False


class TestItScoresTheSliceTheMetricReports:
    """SHIPPED-BUG regression (2026-08-19 Stage B run 1).

    The first version scored candidates on the WHOLE waveform while the results
    table reported ``si_sdr_by_depth`` at depth 2. Different objectives, so the
    monotonicity argument did not transfer, and the "ceiling" landed 0.24-0.37 dB
    BELOW ``no_recursion`` -- a bound that cannot lose, losing. It changed 27/75
    and 31/75 speakers, every one of them for the worse on the reported metric.

    The cause is SI-SDR's scale invariance: it fits a scalar before measuring the
    residual, and which samples you include decides what that scalar becomes.
    Over the whole waveform the bit-exact solo copy dominates and pins it near 1,
    so a level error in the overlap region costs full price. Over the depth-2
    slice the scalar floats and absorbs that error for free. A candidate that
    fixes ``G``'s LEVEL while worsening its SHAPE therefore wins the first
    comparison and loses the second.
    """

    #: Built so the two objectives genuinely DISAGREE, by reproducing the real
    #: geometry: frames 0-1 are depth 1 and are a BIT-EXACT copy of the target in
    #: both estimates (as the solo path always is), which is what pins the
    #: whole-waveform scale factor near 1. Frames 2-3 are depth 2.
    DEPTH = np.array([1, 1, 2, 2])
    TARGET = np.array([4.0, 4.0, 2.0, -2.0])
    #: overlap at HALF level but almost the right shape. The depth-2 slice
    #: rescales that away for free (~34 dB); the whole waveform cannot, because
    #: the exact solo copy holds the scale at 1, so it pays full price (~13 dB).
    CURRENT = np.array([4.0, 4.0, 1.02, -0.98])
    #: overlap at the RIGHT level but a worse shape. Whole waveform ~21 dB
    #: (better), depth-2 slice ~14 dB (worse).
    CANDIDATE = np.array([4.0, 4.0, 2.4, -1.6])

    def test_the_fixture_really_does_pit_the_two_objectives_against_each_other(self):
        """Guards the guard. If whole-waveform did NOT prefer the candidate,
        this test would pass for the wrong reason and protect nothing."""
        from dagger.metrics.sisdr import si_sdr, si_sdr_regionwise

        whole_gain = si_sdr(self.CANDIDATE, self.TARGET) - si_sdr(self.CURRENT, self.TARGET)
        mask = self.DEPTH >= 2
        slice_gain = (si_sdr_regionwise(self.CANDIDATE, self.TARGET, mask)
                      - si_sdr_regionwise(self.CURRENT, self.TARGET, mask))
        assert whole_gain > 0, "fixture broken: whole-waveform does not prefer the candidate"
        assert slice_gain < 0, "fixture broken: the scored slice does not prefer the current"

    def test_a_candidate_better_overall_but_worse_at_depth_2_is_REJECTED(self):
        """The exact decision the shipped version got wrong."""
        accept = make_oracle_accept_fn([self.TARGET], self.DEPTH)
        assert accept(0, self.CANDIDATE, self.CURRENT) is False

    def test_frames_below_min_depth_do_not_influence_the_decision(self):
        """Depth-1 frames are copied from the mixture; the embedding cannot
        change them, so they must not vote on whether it changed for the better."""
        accept = make_oracle_accept_fn([self.TARGET], self.DEPTH)
        verdict = accept(0, self.CANDIDATE, self.CURRENT)

        # Perturb ONLY the unscored frames, arbitrarily hard. Same verdict.
        for scale in (0.0, 10.0, -3.0):
            cand = self.CANDIDATE.copy(); cand[:2] *= scale
            cur = self.CURRENT.copy();    cur[:2] *= scale
            assert accept(0, cand, cur) is verdict, scale

    def test_min_depth_selects_the_slice(self):
        """With min_depth=1 the whole signal is scored and the verdict flips --
        which is precisely the old, wrong behaviour, kept reachable so the
        difference between the two is demonstrable rather than asserted."""
        strict = make_oracle_accept_fn([self.TARGET], self.DEPTH, min_depth=2)
        loose = make_oracle_accept_fn([self.TARGET], self.DEPTH, min_depth=1)
        assert strict(0, self.CANDIDATE, self.CURRENT) is False
        assert loose(0, self.CANDIDATE, self.CURRENT) is True


class TestTheCeilingCannotLose:
    """The property that would have caught the bug, asserted end to end.

    An acceptance rule that only ever accepts improvements cannot produce a
    result worse than accepting nothing -- and accepting nothing is exactly
    ``no_recursion``, since refinement round 0 starts from the enrollment
    embeddings. So the ceiling's deficit against ``no_recursion``, measured on
    the metric it optimizes, must be >= 0.

    Nothing checked that before, which is why a ceiling scoring -0.24 dB shipped
    and was read as a finding about refinement rather than about the objective.
    """

    def test_refined_output_never_scores_worse_than_unrefined(self, fake_encoder):
        from dagger.metrics.sisdr import si_sdr_regionwise
        from dagger.reconstruct.stitch import reconstruct_all

        x, x_O, activity, solo = _scene()
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        variances = np.array([[0.001, 0.001, 0.001], [0.001, 0.001, 0.001]])
        extractor = _AddEmbeddingExtractor()

        # depth: frames 1-3 are the overlap region in `_scene()`.
        depth = np.array([1, 2, 2, 2, 1])
        sources = np.array([[1.0, 1.5, 2.5, 3.5, 0.0],
                            [0.0, 0.5, 1.0, 1.5, 5.0]])

        accept = make_oracle_accept_fn(list(sources), depth)
        refined, _ = refine_embeddings(
            x, x_O, activity, solo, embeddings, variances, extractor,
            fake_encoder, SAMPLE_RATE, rounds=2, accept_fn=accept, **_GATE_KWARGS,
        )

        before = reconstruct_all(x, x_O, activity, solo, embeddings, extractor)
        after = reconstruct_all(x, x_O, activity, solo, refined, extractor)
        mask = depth >= 2
        for i in range(activity.shape[0]):
            lo = si_sdr_regionwise(before[i], sources[i], mask)
            hi = si_sdr_regionwise(after[i], sources[i], mask)
            assert hi >= lo - 1e-9, (
                f"speaker {i}: oracle-gated refinement LOST {lo - hi:.3f} dB on the "
                "metric it optimizes -- the rule and the score disagree"
            )


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
