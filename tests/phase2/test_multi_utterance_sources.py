"""Tests for multi-utterance sources: ``source_k_path`` as ``"pathA|pathB"``.

Experiment 2 (CLAUDE.md Phase 2) needs a speaker's solo region and their
overlapped speech to come from *different recordings*. In stock LibriMix a
speaker contributes one utterance and the scheduler splits it in two, so the
overlap contains nothing about the speaker the solo clip doesn't already --
which caps what embedding refinement could recover, and makes the "refinement
doesn't help" result unfalsifiable in that corpus.

The properties pinned here:

* a ``|``-separated path concatenates, in order, so utterance A is the head and
  therefore the solo slot;
* a plain path still behaves exactly as before;
* with ``min_solo`` <= len(A), the solo region lies entirely inside A -- if it
  spilled into B the two experiment arms would blur together.

``read_wav`` is monkeypatched to synthesize tones, matching
tests/phase2/test_librimix_scheduled.py, so no audio files are needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import dagger.data.librimix as librimix_mod
from dagger.diarize.oracle import activity_matrix, solo_overlap_regions

SAMPLE_RATE = 8000

# Distinct constant amplitudes per utterance, so a concatenation is trivially
# decodable: which utterance a sample came from is readable off its value.
UTTERANCE_VALUES = {"A": 1.0, "B": 2.0, "C": 3.0}
UTTERANCE_SECONDS = {"A": 1.0, "B": 2.0, "C": 1.5}


def _fake_read_wav(path, target_sample_rate):
    key = Path(path).name[0]  # "A.flac" -> "A"
    length = int(UTTERANCE_SECONDS[key] * SAMPLE_RATE)
    return np.full(length, UTTERANCE_VALUES[key], dtype=np.float64)


def _dataset(monkeypatch, tmp_path, *, min_solo_ms: float, n_src: int = 2):
    monkeypatch.setattr(librimix_mod, "resolve_data_root", lambda: tmp_path)
    monkeypatch.setattr(librimix_mod, "read_wav", _fake_read_wav)
    monkeypatch.setattr(librimix_mod, "_resolve_source_path", lambda raw, root: Path(raw))

    ds = object.__new__(librimix_mod.LibriMixDataset)
    ds.sample_rate = SAMPLE_RATE
    ds.n_src = n_src
    ds.overlap = 0.5
    ds.min_solo = int(round(min_solo_ms / 1000.0 * SAMPLE_RATE))
    ds.placement = "scheduled"
    ds.limit = None
    ds.offset = 0
    ds.data_root = tmp_path
    return ds


def _row(paths: list[str]) -> dict:
    row = {"mixture_ID": "fake"}
    for k, path in enumerate(paths, start=1):
        row[f"source_{k}_path"] = path
        row[f"source_{k}_gain"] = "1.0"
    return row


class TestConcatenation:
    def test_pipe_separated_paths_concatenate_in_order(self, monkeypatch, tmp_path):
        ds = _dataset(monkeypatch, tmp_path, min_solo_ms=500.0)

        scene = ds._scene_from_row(_row(["A.flac|B.flac", "C.flac"]))

        source = scene.sources[0]
        active = source[source != 0.0]
        # A (1 s of 1.0) then B (2 s of 2.0) -- order matters: A is the head, so
        # A is what lands in the solo slot.
        assert active[0] == 1.0
        assert np.isclose(np.sum(active == 1.0), UTTERANCE_SECONDS["A"] * SAMPLE_RATE)
        assert np.isclose(np.sum(active == 2.0), UTTERANCE_SECONDS["B"] * SAMPLE_RATE)
        first_b = int(np.argmax(active == 2.0))
        assert np.all(active[:first_b] == 1.0), "B appears before A ends"

    def test_a_single_path_is_unchanged(self, monkeypatch, tmp_path):
        """Every pre-existing metadata CSV must behave exactly as before."""
        ds = _dataset(monkeypatch, tmp_path, min_solo_ms=500.0)

        scene = ds._scene_from_row(_row(["A.flac", "C.flac"]))

        source = scene.sources[0]
        assert np.isclose(np.sum(source == 1.0), UTTERANCE_SECONDS["A"] * SAMPLE_RATE)
        assert not np.any(source == 2.0)


class TestSoloSlotStaysInsideUtteranceA:
    def _solo_values(self, scene, speaker_index: int) -> set[float]:
        activity, _ = activity_matrix(
            scene.segments, num_samples=scene.mixture.shape[0],
            sample_rate=scene.sample_rate, speakers=scene.speakers,
        )
        solo, _ = solo_overlap_regions(activity)
        picked = scene.sources[speaker_index][solo[speaker_index] > 0]
        return set(np.unique(picked).tolist())

    def test_the_taper_tail_is_a_second_solo_region_made_of_b(self, monkeypatch, tmp_path):
        """The non-obvious contamination path, and the reason
        build_heterogeneous_metadata.py hard-errors on a wide duration band.

        The designated slot is not the only place a speaker is alone. In the
        overlap zone, tails end at different times, so whichever speaker outlasts
        the rest is alone for the difference -- a second solo run, made of
        utterance B. `select_topk_solo_clips` takes the LONGEST run, so when the
        taper wins, enrollment reads B and the treatment arm silently becomes the
        control.

        Here speaker 0's designated slot is 0.5 s while its taper runs 1.5 s, so
        B is reachable by enrollment.
        """
        ds = _dataset(monkeypatch, tmp_path, min_solo_ms=500.0)

        scene = ds._scene_from_row(_row(["A.flac|B.flac", "C.flac"]))

        assert UTTERANCE_VALUES["B"] in self._solo_values(scene, 0)

    def test_a_dominant_designated_slot_keeps_enrollment_inside_a(self, monkeypatch, tmp_path):
        """The configuration the generator's taper check guarantees: when the
        designated slot is longer than any taper, the longest solo run -- the one
        enrollment actually reads -- lies inside utterance A."""
        from dagger.enroll.topk import select_topk_solo_clips

        # Both speakers end together (A 1 s + B 2 s vs C 1.5 s placed to match),
        # so no taper exists and the 1 s designated slot is the only solo run.
        ds = _dataset(monkeypatch, tmp_path, min_solo_ms=1000.0)
        monkeypatch.setitem(UTTERANCE_SECONDS, "C", 2.0)

        scene = ds._scene_from_row(_row(["A.flac|B.flac", "C.flac"]))
        activity, _ = activity_matrix(
            scene.segments, num_samples=scene.mixture.shape[0],
            sample_rate=scene.sample_rate, speakers=scene.speakers,
        )
        solo, _ = solo_overlap_regions(activity)
        clips = select_topk_solo_clips(
            scene.sources[0], solo[0], SAMPLE_RATE, k=1, min_clip_ms=100.0
        )

        assert len(clips) == 1
        assert set(np.unique(clips[0]).tolist()) == {UTTERANCE_VALUES["A"]}

    def test_min_solo_longer_than_a_spills_into_b(self, monkeypatch, tmp_path):
        """The failure mode the config comment warns about: with min_solo above
        len(A), enrollment reads audio from B as well, and the treatment and
        control arms stop being distinguishable."""
        ds = _dataset(monkeypatch, tmp_path, min_solo_ms=1500.0)  # A is only 1 s

        scene = ds._scene_from_row(_row(["A.flac|B.flac", "C.flac"]))

        assert UTTERANCE_VALUES["B"] in self._solo_values(scene, 0)
