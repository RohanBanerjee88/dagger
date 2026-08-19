"""Biasing the predicted overlap mask toward inclusion (Phase 3 Stage B item 1).

Stage A attributed the whole oracle-vs-real gap to the activity masks, and the
DER decomposition named the mechanism: ``overlap_recall`` 0.758, i.e. ~24% of
true overlap frames called solo, where the pipeline copies the mixture verbatim
instead of extracting. The costs are asymmetric -- an over-called overlap frame
merely runs ``G`` unnecessarily; an under-called one emits an unseparated
mixture as a speaker's track -- so the fix is a deliberate bias, not a better
estimator.

These tests pin the *invariants* the knob must not break, because the knob sits
directly on the audio path. The one that matters most is the partition of unity:
``w_Ei + w_Oi == activity_i`` is what makes the reconstruction seamless
(CLAUDE.md §2 "soft masks at seams"), and a solo/overlap split that stops being
a partition would produce level errors rather than an exception.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeDiarizer  # noqa: E402

from dagger.diarize.oracle import OracleDiarizer  # noqa: E402
from dagger.diarize.regions import dilate_overlap, scene_regions  # noqa: E402
from dagger.reconstruct.stitch import crossfade_windows  # noqa: E402


@pytest.fixture
def regions(three_speaker_scheduled_scene):
    return scene_regions(three_speaker_scheduled_scene, OracleDiarizer())


class TestDilationIsMonotone:
    def test_zero_is_the_identity(self, regions):
        """A sweep's zero point must be the undilated baseline exactly."""
        assert dilate_overlap(regions, 0) is regions

    def test_overlap_only_grows(self, regions):
        for samples in (1, 10, 100):
            widened = dilate_overlap(regions, samples)
            assert np.all(widened.overlap >= regions.overlap), samples

    def test_solo_only_shrinks(self, regions):
        """Frames leave solo for overlap, never the reverse."""
        for samples in (1, 10, 100):
            widened = dilate_overlap(regions, samples)
            assert np.all(widened.solo <= regions.solo), samples

    def test_growth_is_bounded_by_the_half_width(self, regions):
        """A dilation of k samples adds at most k frames per boundary.

        Guards against an off-by-one that would make the config key mean
        something other than what its name says.
        """
        samples = 5
        widened = dilate_overlap(regions, samples)
        boundaries = int(np.count_nonzero(np.diff(regions.overlap)))
        added = int(widened.overlap.sum() - regions.overlap.sum())
        assert added <= boundaries * samples

    def test_negative_is_rejected(self, regions):
        with pytest.raises(ValueError, match="samples >= 0"):
            dilate_overlap(regions, -1)

    @pytest.mark.parametrize("scale", [1, 2, 10])
    def test_a_dilation_wider_than_the_scene_still_returns_the_scene(
        self, regions, scale
    ):
        """Shipped-bug regression: ``np.convolve(mode="same")`` overflows.

        It returns ``max(len(signal), len(kernel))`` samples, so a half-width
        past the scene length produced an over-long mask -- which happened to
        raise on the broadcast against ``activity`` here, but would have
        silently mis-sized a squarer array. The sweep reaches this regime
        legitimately: a large dilation value is exactly how the enrollment
        starvation limit gets measured.
        """
        n = regions.overlap.shape[0]
        widened = dilate_overlap(regions, n * scale)
        assert widened.overlap.shape == (n,)
        assert widened.solo.shape == regions.solo.shape
        # Everything active is now overlap, so nothing is left to enroll from.
        assert widened.overlap.all()
        assert not widened.solo.any()


class TestInvariantsTheAudioPathRelieson:
    def test_partition_of_unity_still_holds(self, regions):
        """``w_Ei + w_Oi == activity_i`` at every sample, dilated or not.

        This is the property that makes the solo<->overlap seam click-free. A
        broken partition would not raise -- it would quietly scale a speaker's
        output, which is exactly the class of defect that survives a real run.
        """
        widened = dilate_overlap(regions, 20)
        for i in range(widened.num_speakers):
            w_solo, w_over = crossfade_windows(widened.solo[i], widened.activity[i], fade=8)
            np.testing.assert_allclose(w_solo + w_over, widened.activity[i], atol=1e-12)

    def test_solo_stays_a_subset_of_activity(self, regions):
        """``enroll_speaker``'s contamination guard depends on this.

        ``select_topk_solo_clips`` raises a plain ValueError (deliberately NOT
        the catchable NoSoloRegionError) when solo is not a subset of activity,
        because that signals a region bug rather than benign missing audio.
        """
        widened = dilate_overlap(regions, 50)
        for i in range(widened.num_speakers):
            assert np.all(widened.solo[i].astype(bool) <= widened.activity[i].astype(bool))

    def test_activity_and_depth_are_untouched(self, regions):
        """Depth is the stratification axis and must not move with an audio knob.

        If dilation could change ``depth``, the knob would relabel which bucket
        each sample is scored in -- letting it grade itself on a curve it drew.
        """
        widened = dilate_overlap(regions, 40)
        np.testing.assert_array_equal(widened.activity, regions.activity)
        np.testing.assert_array_equal(widened.depth, regions.depth)
        assert widened.speakers == regions.speakers


class TestItRecoversMissedOverlap:
    def test_a_shrunken_overlap_mask_is_restored(self, three_speaker_scheduled_scene):
        """The actual point: dilation recovers frames a diarizer under-called.

        Simulates the measured failure by eroding the true overlap mask, then
        checks dilation puts the lost frames back on the extract path.
        """
        truth = scene_regions(three_speaker_scheduled_scene, OracleDiarizer())
        eroded = truth.overlap.copy()
        run = np.flatnonzero(eroded)
        assert run.size > 40, "fixture needs a usable overlap run"
        eroded[run[:20]] = 0.0  # diarizer called the leading 20 samples "solo"

        from dagger.diarize.regions import Regions

        shrunk = Regions(
            activity=truth.activity, speakers=truth.speakers,
            solo=truth.activity * (1.0 - eroded)[None, :],
            overlap=eroded, depth=truth.depth,
        )
        missed_before = int(np.sum((truth.overlap > 0) & (shrunk.overlap == 0)))
        assert missed_before == 20

        widened = dilate_overlap(shrunk, 20)
        missed_after = int(np.sum((truth.overlap > 0) & (widened.overlap == 0)))
        assert missed_after == 0

    def test_real_diarizer_regions_dilate_without_error(
        self, three_speaker_scheduled_scene
    ):
        """The knob must survive a discovered-cluster row set, not just oracle's."""
        predicted = scene_regions(three_speaker_scheduled_scene, FakeDiarizer())
        widened = dilate_overlap(predicted, 25)
        assert widened.overlap.sum() >= predicted.overlap.sum()
        assert widened.solo.shape == predicted.solo.shape
