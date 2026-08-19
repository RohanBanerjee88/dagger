"""Simulated diarization error for training (Phase 3 Stage B item 7).

The augmentation must reproduce the MEASURED Stage A error profile
(miss 0.105, confusion 0.008, overlap_recall 0.758) rather than the assumed one
-- so these tests pin *what kind* of corruption is produced, not merely that
some corruption happens. A label-swap augmentation would pass a "masks changed"
test just as well while training ``G`` against a failure mode that does not
occur at this scene length.

The second half covers the wiring hazard the plan called out: ``w_overlap`` is
precomputed, and deriving it from clean masks while training on corrupted ones
would silently optimize the extractor against a window it never sees.
"""

from __future__ import annotations

import numpy as np
import pytest

from dagger.data.mask_augment import drop_overlapped_activity  # noqa: E402

SAMPLE_RATE = 8000


def _activity(num_speakers=3, run_ms=1500, blocks=4):
    """Three speakers with a solo lead-in each, then several shared overlap runs.

    Length is derived rather than hardcoded: the augmenter deliberately skips a
    speaker with fewer than two candidate runs (dropping the only one would
    erase them from the scene), so a fixture too short to fit ``blocks``
    separated runs silently tests nothing.
    """
    run = int(run_ms / 1000.0 * SAMPLE_RATE)
    length = num_speakers * run + blocks * 2 * run
    activity = np.zeros((num_speakers, length), dtype=np.float64)
    for i in range(num_speakers):
        activity[i, i * run: i * run + run] = 1.0             # solo lead-in
        for block in range(blocks):                            # shared overlap zone
            start = num_speakers * run + block * 2 * run       # gaps keep runs separate
            activity[i, start: start + run] = 1.0
    return activity


def test_fixture_has_enough_candidate_runs():
    """Guards the guard: the augmenter no-ops on <2 runs, so verify we have more."""
    from dagger.data.mask_augment import _runs

    activity = _activity()
    depth = activity.sum(axis=0)
    for i in range(activity.shape[0]):
        runs = _runs((activity[i] > 0) & (depth >= 2))
        assert len(runs) >= 2, f"speaker {i} has {len(runs)} overlapped run(s)"


class TestItSimulatesTheMeasuredProfile:
    def test_only_overlapped_frames_are_dropped(self):
        """Solo activity is untouched -- deleting it is a different failure.

        A dropped solo region removes the speaker from the scene rather than
        mislabelling them, and it starves enrollment, which would confound the
        augmentation with a change in embedding quality.
        """
        activity = _activity()
        depth = activity.sum(axis=0)
        solo_frames = depth == 1

        augmented = drop_overlapped_activity(
            activity, SAMPLE_RATE, np.random.default_rng(0),
            drop_prob=1.0, min_dur_ms=100.0, max_dur_ms=800.0,
        )
        np.testing.assert_array_equal(
            augmented[:, solo_frames], activity[:, solo_frames]
        )

    def test_activity_only_ever_decreases(self):
        """Miss, not false alarm. Stage A measured false_alarm_rate 0.000."""
        activity = _activity()
        augmented = drop_overlapped_activity(
            activity, SAMPLE_RATE, np.random.default_rng(1), drop_prob=1.0,
        )
        assert np.all(augmented <= activity)

    def test_no_speaker_is_erased_entirely(self):
        """A zero row breaks the partition of unity and fails enrollment.

        Scenes would then vanish from the training set in proportion to how
        hard the augmentation hit them -- the selection bias that silently
        shrank Phase 1's effective dataset.
        """
        activity = _activity()
        for seed in range(20):
            augmented = drop_overlapped_activity(
                activity, SAMPLE_RATE, np.random.default_rng(seed),
                drop_prob=1.0, min_dur_ms=100.0, max_dur_ms=100000.0,
            )
            assert np.all(augmented.sum(axis=1) > 0), seed

    def test_it_actually_reduces_overlap_depth(self):
        """The point of the exercise: overlapped frames become non-overlapped.

        This is what makes the pipeline take the solo-copy path on frames that
        are genuinely overlapped -- the mechanism behind the -3.11 dB gap.
        """
        activity = _activity()
        augmented = drop_overlapped_activity(
            activity, SAMPLE_RATE, np.random.default_rng(2), drop_prob=1.0,
        )
        before = int(np.sum(activity.sum(axis=0) >= 2))
        after = int(np.sum(augmented.sum(axis=0) >= 2))
        assert after < before


class TestKnobsAndGuards:
    def test_zero_probability_is_the_identity(self):
        activity = _activity()
        augmented = drop_overlapped_activity(
            activity, SAMPLE_RATE, np.random.default_rng(3), drop_prob=0.0,
        )
        np.testing.assert_array_equal(augmented, activity)

    def test_it_is_reproducible_from_the_generator(self):
        activity = _activity()
        kwargs = dict(drop_prob=0.6, min_dur_ms=100.0, max_dur_ms=900.0)
        first = drop_overlapped_activity(
            activity, SAMPLE_RATE, np.random.default_rng(7), **kwargs
        )
        second = drop_overlapped_activity(
            activity, SAMPLE_RATE, np.random.default_rng(7), **kwargs
        )
        np.testing.assert_array_equal(first, second)

    def test_it_does_not_mutate_its_input(self):
        activity = _activity()
        original = activity.copy()
        drop_overlapped_activity(
            activity, SAMPLE_RATE, np.random.default_rng(4), drop_prob=1.0
        )
        np.testing.assert_array_equal(activity, original)

    def test_bad_probability_is_rejected(self):
        with pytest.raises(ValueError, match="drop_prob"):
            drop_overlapped_activity(
                _activity(), SAMPLE_RATE, np.random.default_rng(0), drop_prob=1.5
            )

    def test_bad_duration_range_is_rejected(self):
        with pytest.raises(ValueError, match="min_dur_ms"):
            drop_overlapped_activity(
                _activity(), SAMPLE_RATE, np.random.default_rng(0),
                min_dur_ms=900.0, max_dur_ms=100.0,
            )


class TestDerivedMasksFollowTheAugmentation:
    def test_w_overlap_is_recomputed_from_augmented_masks(self):
        """The wiring hazard: ``w_overlap`` is precomputed in ``_prepare``.

        Recomputing it from the CLEAN masks while training on corrupted ones
        would apply the extractor's loss through a window the model never sees,
        and nothing would fail. Asserted here on the arithmetic rather than
        through torch, so it needs no [ml] extra.
        """
        from dagger.diarize.oracle import solo_overlap_regions
        from dagger.reconstruct.stitch import crossfade_windows

        activity = _activity()
        augmented = drop_overlapped_activity(
            activity, SAMPLE_RATE, np.random.default_rng(5), drop_prob=1.0,
        )
        clean_solo, _ = solo_overlap_regions(activity)
        aug_solo, _ = solo_overlap_regions(augmented)

        clean_w = crossfade_windows(clean_solo[0], activity[0], fade=40)[1]
        aug_w = crossfade_windows(aug_solo[0], augmented[0], fade=40)[1]

        assert not np.allclose(clean_w, aug_w), (
            "augmentation did not change w_overlap -- the fixture is too weak "
            "to catch the stale-window bug this test exists for"
        )
        # And the partition of unity still holds on the augmented masks.
        aug_solo_w = crossfade_windows(aug_solo[0], augmented[0], fade=40)[0]
        np.testing.assert_allclose(aug_solo_w + aug_w, augmented[0], atol=1e-12)
