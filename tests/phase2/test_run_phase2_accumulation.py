"""Tests for scripts/run_phase2.py's accumulation counter.

``_accepted_before`` turns the speaker-indexed accept sequence returned by
:func:`dagger.reconstruct.deflation.reconstruct_all_deflation` into "how many
prior estimates were subtracted into the residual before speaker i was
extracted" -- the counter Theorem 3's ``L*||E_(m-1)||`` penalty is indexed by,
and the only column that mechanically separates gated from ungated deflation.

The helper is pure (a permutation and a bool array in, a list of ints out), so
these tests need no corpus, GPU, encoder, or extractor -- only the module,
loaded the same way tests/phase2/test_train_phase1_curriculum.py loads its
script under test.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def _load_run_phase2():
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "run_phase2_under_test", repo_root / "scripts" / "run_phase2.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_phase2 = _load_run_phase2()


class TestAcceptedBefore:
    def test_ungated_accumulation_equals_position_in_order(self):
        """Ungated deflation accepts unconditionally, so every speaker inherits
        exactly as many prior estimates as its position in ``order`` -- this is
        the invariant score_scene asserts at runtime."""
        order = [2, 0, 3, 1]
        accepts = np.ones(4, dtype=bool)

        counts = run_phase2._accepted_before(order, accepts)

        # Speaker 2 goes first (0 priors), then 0, then 3, then 1.
        assert counts == [1, 3, 0, 2]

    def test_rejections_do_not_contaminate_later_speakers(self):
        """A rejected estimate leaves the residual untouched, so it must not be
        counted against anyone extracted after it."""
        order = [0, 1, 2, 3]
        accepts = np.array([True, False, True, False])

        counts = run_phase2._accepted_before(order, accepts)

        # Speaker 1's rejection means speaker 2 still only inherits speaker 0's
        # contribution, and speaker 3 inherits speakers 0 and 2's.
        assert counts == [0, 1, 1, 2]

    def test_all_rejected_means_no_accumulation_anywhere(self):
        """A gate that rejects everything makes gated deflation degenerate into
        no_recursion: every speaker extracts from a pristine x_O."""
        order = [1, 0, 2]
        accepts = np.zeros(3, dtype=bool)

        assert run_phase2._accepted_before(order, accepts) == [0, 0, 0]

    def test_counter_is_indexed_by_speaker_not_by_order_position(self):
        """The returned list is speaker-indexed. With a non-identity order and
        exactly one acceptance, only speakers extracted *after* the accepted one
        inherit anything -- indexing the result by order position instead would
        attribute the accumulation to the wrong speakers."""
        order = [2, 1, 0]
        accepts = np.array([False, True, False])  # only speaker 1 accepted

        counts = run_phase2._accepted_before(order, accepts)

        assert counts[2] == 0  # extracted first, nothing before it
        assert counts[1] == 0  # extracted second; speaker 2 was rejected
        assert counts[0] == 1  # extracted last, after speaker 1 was accepted


class TestFieldSchema:
    def test_gate_columns_are_not_in_the_score_rows(self):
        """Gate decisions are per-speaker but score rows are per (speaker,
        depth), so a gate column in the score file would be duplicated across
        however many depths a speaker spans -- inflating any accept rate
        counted from it. The two grains stay in separate files."""
        assert not [f for f in run_phase2.SCORE_FIELDS if f.startswith("gate")]
        assert "round" in run_phase2.GATE_FIELDS

    def test_deflation_systems_are_named_explicitly(self):
        """Membership is a fixed tuple rather than a `endswith("deflation")`
        name test, so a future system can't be silently opted in."""
        assert run_phase2.DEFLATION_SYSTEMS == ("ungated_deflation", "gated_deflation")
        assert all(s in run_phase2.SYSTEMS for s in run_phase2.DEFLATION_SYSTEMS)
