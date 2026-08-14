"""The Phase 3 seam: region derivation, the oracle equivalence, and leakage guards.

Three separate claims are pinned here, and each one is a claim the plan asserted
in prose. Prose is not insurance — every reporting defect this project shipped
survived a real run precisely because *nothing failed* when it was present.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeDiarizer, build_scheduled_scene  # noqa: E402

from dagger.diarize.oracle import (  # noqa: E402
    OracleDiarizer,
    activity_matrix,
    overlap_depth,
    solo_overlap_regions,
)
from dagger.diarize.regions import scene_regions  # noqa: E402
from dagger.eval.systems import deflation_order  # noqa: E402

SAMPLE_RATE = 8000


class TestOracleRegionsAreUnchanged:
    """`scene_regions(scene, OracleDiarizer())` must equal the old inline code.

    This is the mechanical half of the A5 refactor guard. The other half — that
    `run_phase2.py` still reproduces a committed CSV byte-identically — needs the
    corpus and a GPU; this needs neither, and it is the part that would silently
    drift.
    """

    def test_matches_the_pre_refactor_inline_derivation(self, three_speaker_scheduled_scene):
        scene = three_speaker_scheduled_scene

        # Exactly what scripts/run_phase2.py did before the move.
        n = scene.mixture.shape[0]
        activity, speakers = activity_matrix(
            scene.segments, num_samples=n, sample_rate=scene.sample_rate,
            speakers=scene.speakers,
        )
        solo, overlap = solo_overlap_regions(activity)
        depth = overlap_depth(activity)

        regions = scene_regions(scene, OracleDiarizer())
        assert regions.speakers == speakers
        np.testing.assert_array_equal(regions.activity, activity)
        np.testing.assert_array_equal(regions.solo, solo)
        np.testing.assert_array_equal(regions.overlap, overlap)
        np.testing.assert_array_equal(regions.depth, depth)

    def test_a_speaker_with_no_segments_still_gets_a_row(self):
        """The scene builders skip zero-length chunks, so a silent speaker emits
        no segments at all. Binding the oracle's label set keeps its all-zero row
        — without which row `i` would stop meaning source `i` and every
        downstream index would shift by one, silently.
        """
        scene = build_scheduled_scene(
            lengths=[SAMPLE_RATE * 2, 0, SAMPLE_RATE * 2], min_solo=SAMPLE_RATE
        )
        assert all(seg.speaker != "s2" for seg in scene.segments)

        regions = scene_regions(scene, OracleDiarizer())
        assert regions.speakers == ["s1", "s2", "s3"]
        assert regions.activity.shape[0] == 3
        assert regions.activity[1].sum() == 0


class TestRealRegionsDiscoverClusters:
    def test_extra_cluster_survives_instead_of_vanishing(self, three_speaker_scheduled_scene):
        """`activity_matrix` silently drops labels it was not told about.

        Binding the real arm to `scene.speakers` would therefore erase an
        invented speaker without a trace — turning a diarization failure into no
        evidence of a problem. Discovery is what keeps it visible.
        """
        regions = scene_regions(
            three_speaker_scheduled_scene, FakeDiarizer(extra_cluster=True)
        )
        assert regions.activity.shape[0] == 4
        assert "SPEAKER_99" in regions.speakers

    def test_dropped_speaker_yields_fewer_rows(self, three_speaker_scheduled_scene):
        regions = scene_regions(three_speaker_scheduled_scene, FakeDiarizer(drop_speaker=1))
        assert regions.activity.shape[0] == 2

    def test_predicted_rows_are_not_ground_truth_labels(self, three_speaker_scheduled_scene):
        regions = scene_regions(three_speaker_scheduled_scene, FakeDiarizer())
        assert regions.speakers != three_speaker_scheduled_scene.speakers
        assert all(s.startswith("SPEAKER_") for s in regions.speakers)


class TestDeflationOrderPolicy:
    """The confound fix. See dagger.eval.systems.deflation_order."""

    def test_variance_policy_is_index_order_when_variance_is_zero(self):
        """Why the sort has never actually done anything under oracle regions:
        V_i is identically 0, and Python's sort is stable."""
        variances = np.zeros((4, 8))
        assert deflation_order(variances, "variance") == [0, 1, 2, 3]

    def test_variance_policy_reorders_once_variance_is_real(self):
        """And why it starts mattering the moment real diarization fragments
        solo regions: the best-enrolled speaker is promoted to position 0, the
        least-damaged accumulation slot."""
        variances = np.array([[0.9], [0.1], [0.5]])
        assert deflation_order(variances, "variance") == [1, 2, 0]

    def test_index_policy_ignores_variance(self):
        variances = np.array([[0.9], [0.1], [0.5]])
        assert deflation_order(variances, "index") == [0, 1, 2]

    def test_unknown_policy_raises(self):
        with pytest.raises(ValueError, match="deflation.order"):
            deflation_order(np.zeros((2, 2)), "whatever")


class TestScoringTargetsNeverLeakIntoTheAudioPath:
    """Ground truth may decide what an output is COMPARED to, nothing else.

    At inference on real data there are no true labels, so if the extractor, the
    gate, the enrollment or the deflation order could see them, the measured gap
    would be optimistic by an unknown amount.
    """

    def test_regions_carry_no_reference_information(self, three_speaker_scheduled_scene):
        """Whatever the audio path consumes is a function of the diarizer alone."""
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, FakeDiarizer(jitter=0.05))

        # Rebuild from the diarizer's own output, with no access to the Scene's
        # sources or speaker labels beyond what the diarizer itself emitted.
        segments = FakeDiarizer(jitter=0.05).diarize(scene)
        activity, speakers = activity_matrix(
            segments, num_samples=scene.mixture.shape[0], sample_rate=scene.sample_rate
        )
        np.testing.assert_array_equal(regions.activity, activity)
        assert regions.speakers == speakers

    def test_score_targets_is_identity_under_oracle(self, three_speaker_scheduled_scene):
        from dagger.eval.systems import _score_targets

        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, OracleDiarizer())
        targets = _score_targets(scene, regions)
        assert len(targets) == len(scene.speakers)
        for i, (label, target) in enumerate(targets):
            assert label == scene.speakers[i]
            np.testing.assert_array_equal(target, scene.sources[i])

    def test_score_targets_undoes_the_cluster_permutation(self, three_speaker_scheduled_scene):
        """A permuted diarizer must not produce mis-attributed scores.

        Row `i` of the output is cluster `i`; the target it is compared against
        has to be whichever source that cluster actually corresponds to.
        """
        from dagger.eval.systems import _score_targets

        scene = three_speaker_scheduled_scene
        permutation = [2, 0, 1]
        regions = scene_regions(scene, FakeDiarizer(permutation=permutation))
        targets = _score_targets(scene, regions)

        # FakeDiarizer labels speakers[permutation[rank]] as SPEAKER_{rank}, and
        # activity rows follow first-appearance order of the emitted segments.
        for row, cluster in enumerate(regions.speakers):
            rank = int(cluster.split("_")[1])
            expected_source = permutation[rank]
            label, target = targets[row]
            # The row must be LABELLED with the ground-truth speaker, not the
            # cluster id -- that label is what paired comparisons key on.
            assert label == scene.speakers[expected_source]
            np.testing.assert_array_equal(target, scene.sources[expected_source])

    def test_unattributable_cluster_scores_against_nothing(self, three_speaker_scheduled_scene):
        from dagger.eval.systems import _score_targets

        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, FakeDiarizer(extra_cluster=True))
        targets = _score_targets(scene, regions)
        # The phantom speaker has no ground-truth counterpart, so there is
        # nothing to score it against -- None, not a silently wrong source.
        assert sum(1 for label, target in targets if target is None) == 1
        assert sum(1 for label, target in targets if label is None) == 1
