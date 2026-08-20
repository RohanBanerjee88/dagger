"""Scores are bucketed by TRUE concurrent-speaker count, not the diarizer's belief.

Depth is defined as intrinsic difficulty (CLAUDE.md §6.4) — a property of the
acoustic scene. Bucketing by *predicted* depth lets a diarizer grade itself on a
curve it drew: merge three concurrent speakers into one cluster and every sample
of a genuine 3-way overlap is filed under "depth 1", the easy row. That is
exactly what the 2026-08-16 run did — 70% of every scene was 3-way overlap and
the real arm reported no depth-3 rows at all.

It also breaks the cross-arm pairing, which keys on `depth`: if `depth 2` names a
different set of samples in each arm, their difference is not a controlled
comparison.

None of this hands the method the speaker count. The audio path runs on predicted
regions throughout and emits however many tracks the diarizer found; only the
reporting bucket comes from the reference — the same choice DER already makes for
its denominator.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeDiarizer  # noqa: E402

from dagger.diarize.oracle import OracleDiarizer, overlap_depth  # noqa: E402
from dagger.diarize.regions import scene_regions  # noqa: E402
from dagger.eval.systems import score_scene  # noqa: E402

from test_run_phase3_arms import (  # noqa: E402
    GATE_CFG,
    _DeterministicExtractor,
    _FakeEncoder,
)


class _MergingDiarizer(FakeDiarizer):
    """Collapses every speaker into ONE cluster spanning the whole scene.

    The pathological case observed for real: predicted depth never exceeds 1, so
    predicted-depth bucketing would report the entire scene as solo.
    """

    binds_scene_speakers = False

    def diarize(self, scene):
        from dagger.diarize.oracle import Segment

        duration = scene.mixture.shape[0] / scene.sample_rate
        return [Segment(speaker="SPEAKER_00", start=0.0, duration=duration)]


def _run(scene, diarizer, **kw):
    regions = scene_regions(scene, diarizer)
    return score_scene(
        scene, 0, 3, 100.0, None, _FakeEncoder(), _DeterministicExtractor(),
        GATE_CFG, 0, regions=regions, on_unenrollable="drop", **kw
    )


class TestDepthComesFromTheReference:
    def test_a_merged_diarization_still_reports_true_depths(
        self, three_speaker_scheduled_scene
    ):
        scene = three_speaker_scheduled_scene
        oracle_depths = set(np.unique(
            overlap_depth(scene_regions(scene, OracleDiarizer()).activity)
        ).tolist())
        assert 3 in oracle_depths, "fixture must contain a genuine 3-way overlap"

        merged = scene_regions(scene, _MergingDiarizer())
        assert int(merged.depth.max()) == 1, "fixture must collapse to one cluster"

        rows, _, _ = _run(scene, _MergingDiarizer())
        depths = {int(r["depth"]) for r in rows}
        # Bucketed by the TRUE count, so the hard samples stay in the hard row
        # even though the diarizer believed the whole scene was solo.
        assert 3 in depths, (
            "depth-3 samples vanished -- scores are being bucketed by the "
            "diarizer's belief, so it is grading itself on its own curve"
        )

    def test_buckets_match_the_oracle_arm_exactly(self, three_speaker_scheduled_scene):
        """The precondition for pairing: `depth 2` must name the same samples
        in every arm, or the cross-arm difference compares different audio."""
        scene = three_speaker_scheduled_scene
        oracle_rows, _, _ = _run(scene, OracleDiarizer())
        real_rows, _, _ = _run(scene, FakeDiarizer(jitter=0.05))

        oracle_depths = {int(r["depth"]) for r in oracle_rows}
        real_depths = {int(r["depth"]) for r in real_rows}
        assert real_depths <= oracle_depths
        assert real_depths, "the real arm produced no scored rows at all"


class TestTheAudioPathIsStillDiarizerDriven:
    """The change must not smuggle the speaker count into the method."""

    def test_output_track_count_still_comes_from_the_diarizer(
        self, three_speaker_scheduled_scene
    ):
        scene = three_speaker_scheduled_scene
        rows, _, _ = _run(scene, _MergingDiarizer())
        # One cluster found -> one track reconstructed, even though the reference
        # says three speakers. Ground truth changed the BUCKET, not the method.
        assert {int(r["m"]) for r in rows} == {1}
        assert {int(r["n_clusters"]) for r in rows} == {1}

    def test_n_clusters_and_m_expose_the_enrollment_drop(
        self, three_speaker_scheduled_scene
    ):
        """`n_clusters` is pre-drop, `m` is post-drop; the pair tells the story.

        Reading this off the 2026-08-16 run took a forensic pass over the CSV.
        """
        scene = three_speaker_scheduled_scene
        rows, _, _ = _run(scene, FakeDiarizer(extra_cluster=True))
        n_clusters = {int(r["n_clusters"]) for r in rows}
        m = {int(r["m"]) for r in rows}
        assert n_clusters == {4}, "the invented cluster should be counted"
        # The phantom cluster sits inside another speaker's speech, so it has no
        # solo region and cannot be enrolled -- dropped, and visibly so.
        assert m == {3}, f"expected the phantom to be dropped, got m={m}"


class TestOracleArmIsUnaffected:
    def test_predicted_and_reference_depth_coincide_under_oracle(
        self, three_speaker_scheduled_scene
    ):
        """Why Phase 2's committed CSVs cannot move: under oracle diarization the
        predicted activity IS the reference activity."""
        scene = three_speaker_scheduled_scene
        regions = scene_regions(scene, OracleDiarizer())
        reference = scene_regions(scene, OracleDiarizer()).activity
        np.testing.assert_array_equal(regions.depth, overlap_depth(reference))
