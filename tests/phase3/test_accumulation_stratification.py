"""The oracle-vs-real accumulation table must not be a biased subset.

A SHIPPED-BUG regression test (found 2026-08-18, Phase 3 Stage A result note).

``scripts/aggregate_phase3.py::_table`` filtered rows on ``x_field`` *before*
pairing them. On the ``depth`` axis that is harmless -- depth comes from the
reference activity for both arms (``dagger.eval.systems``), so the two arms
always agree. On the ``n_accepted_before`` axis it is fatal: deflation order is
ascending ``V_i``, which is identically 0 under oracle diarization but a real,
data-dependent permutation under a real diarizer. So a speaker survived the
filter only when its accumulation position happened to *match* across arms --
which is precisely the thing being measured.

Observed cost on the real Stage A run: n = 34/28/16 for the ``real - oracle``
accumulation table, against n = 98/92/92 for ``real_index_order - oracle``,
where the order matches oracle by construction. The table rendered, the numbers
looked plausible, and nothing failed -- the shape of every reporting defect this
project has shipped.

The fix pairs first and buckets each surviving pair by the REFERENCE arm's
position, which is also the only reading that means anything: "for a speaker the
oracle placed at accumulation level k, what did real diarization cost it?"
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script(name: str):
    repo_root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        f"{name}_under_test", repo_root / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregate_phase3 = _load_script("aggregate_phase3")


def _row(scene, speaker, arm, si_sdr, *, n_accepted_before, depth=2, m=3,
         system="ungated_deflation", dilate_ms=0.0):
    """One typed score row, shaped like ``load_score_rows`` output.

    ``dilate_ms`` is part of the pairing key (an arm difference must be taken at
    one dilation), and ``load_score_rows`` supplies 0.0 for any CSV predating
    the sweep -- so it is always present on a real row.
    """
    return {
        "source": "test.csv",
        "scene": scene,
        "speaker": speaker,
        "system": system,
        "si_sdr": si_sdr,
        "diarization": arm,
        "dilate_ms": dilate_ms,
        "m": m,
        "depth": depth,
        "deflation_index": n_accepted_before,
        "n_accepted_before": n_accepted_before,
        "refine_rounds": 0,
    }


def _reordered_corpus():
    """Two scenes where the real arm's ``V_i`` sort permutes the deflation order.

    Oracle places (a, b, c) at levels (0, 1, 2); real places them at (2, 0, 1).
    Every speaker is scored in BOTH arms, so a correct stratification keeps all
    six pairs -- the levels simply move.
    """
    rows = []
    for scene in ("scene1", "scene2"):
        for speaker, oracle_level, real_level in (
            ("a", 0, 2),
            ("b", 1, 0),
            ("c", 2, 1),
        ):
            # Real is uniformly 2 dB worse, so the expected answer is exactly
            # -2.00 at every level and any bucketing error shows up as a
            # missing row rather than as a shifted mean.
            rows.append(_row(scene, speaker, "oracle", 5.0, n_accepted_before=oracle_level))
            rows.append(_row(scene, speaker, "real", 3.0, n_accepted_before=real_level))
    return rows


def _n_values(lines: list[str]) -> list[int]:
    """The ``n`` column from every data row of a rendered markdown table."""
    counts = []
    for line in lines:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) >= 3 and cells[0] in aggregate_phase3.SYSTEMS:
            counts.append(int(cells[2]))
    return counts


class TestAccumulationStratificationPairsBeforeFiltering:
    def test_every_paired_speaker_is_counted(self):
        """The bug: pairing after filtering drops every reordered speaker.

        All six (scene, speaker) pairs are scored in both arms, so all six must
        appear. Filtering first finds zero rows whose level agrees across arms
        and renders the "no paired rows" placeholder instead.
        """
        lines = aggregate_phase3._table(
            _reordered_corpus(), "real", "oracle", "n_accepted_before"
        )

        assert sum(_n_values(lines)) == 6, (
            "accumulation table lost rows to the pre-pairing filter:\n"
            + "\n".join(lines)
        )

    def test_buckets_use_the_reference_arms_position(self):
        """Levels come from the RIGHT (reference) arm, not the left one.

        Both arms cover levels 0/1/2 here, so a wrong choice of reference is
        invisible in the counts -- it is only detectable by which speaker lands
        in which bucket. Oracle puts speaker 'a' at level 0; real puts it at 2.
        """
        rows = _reordered_corpus()
        # Make speaker 'a' distinguishable: its real-arm score is 4 dB worse
        # rather than 2, so the level holding 'a' has a mean of -3.00 (one 'a'
        # row at -4 averaged with one non-'a' row at -2 per level... except
        # each level holds exactly one speaker per scene, so it is -4.00 flat).
        for row in rows:
            if row["speaker"] == "a" and row["diarization"] == "real":
                row["si_sdr"] = 1.0

        lines = aggregate_phase3._table(rows, "real", "oracle", "n_accepted_before")
        by_level = {}
        for line in lines:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) >= 4 and cells[0] in aggregate_phase3.SYSTEMS:
                by_level[int(cells[1])] = float(cells[3])

        # Oracle placed 'a' at level 0, so level 0 carries the -4 dB rows.
        assert by_level[0] == -4.00, f"level 0 should hold speaker 'a': {by_level}"
        assert by_level[1] == -2.00
        assert by_level[2] == -2.00

    def test_depth_axis_is_unchanged(self):
        """Depth already paired correctly and must not move.

        Depth is derived from the REFERENCE activity for both arms, so the two
        never disagree on it. If this starts failing, the fix reached further
        than intended.
        """
        lines = aggregate_phase3._table(_reordered_corpus(), "real", "oracle", "depth")
        assert sum(_n_values(lines)) == 6
        means = [
            float([c.strip() for c in line.split("|")[1:-1]][3])
            for line in lines
            if len([c.strip() for c in line.split("|")[1:-1]]) >= 4
            and [c.strip() for c in line.split("|")[1:-1]][0] in aggregate_phase3.SYSTEMS
        ]
        assert means == [-2.00]

    def test_arms_never_pair_across_different_dilations(self):
        """A swept CSV holds several pipelines; an arm difference needs one.

        Pairing `real` at 50 ms against `oracle` at 0 ms would measure the
        dilation knob and the diarizer together and report the sum as the
        diarizer's cost -- a plausible-looking number with two variables in it.
        """
        rows = []
        for speaker, level in (("a", 0), ("b", 1), ("c", 2)):
            # oracle only at 0 ms, real only at 50 ms: nothing may pair.
            rows.append(_row("s", speaker, "oracle", 5.0,
                             n_accepted_before=level, dilate_ms=0.0))
            rows.append(_row("s", speaker, "real", 3.0,
                             n_accepted_before=level, dilate_ms=50.0))

        lines = aggregate_phase3._table(rows, "real", "oracle", "n_accepted_before")
        assert sum(_n_values(lines)) == 0, "paired across dilation values"

    def test_matched_dilations_still_pair(self):
        """The complement: same dilation on both arms pairs normally."""
        rows = []
        for speaker, level in (("a", 0), ("b", 1), ("c", 2)):
            for arm, score in (("oracle", 5.0), ("real", 3.0)):
                rows.append(_row("s", speaker, arm, score,
                                 n_accepted_before=level, dilate_ms=50.0))

        lines = aggregate_phase3._table(rows, "real", "oracle", "n_accepted_before")
        assert sum(_n_values(lines)) == 3

    def test_accumulation_free_systems_are_still_excluded(self):
        """``n_accepted_before`` is blank for no_recursion/coarse_to_fine.

        Blank is not zero (``_OPTIONAL_INT_FIELDS`` in phase2_scores): 0 means
        "nothing was deflated before me", which is a different statement from
        "this system never deflates". Folding None into a zero bucket would
        invent an accumulation level for a system that has none.
        """
        rows = _reordered_corpus()
        for scene in ("scene1", "scene2"):
            for speaker in ("a", "b", "c"):
                for arm, score in (("oracle", 5.0), ("real", 3.0)):
                    rows.append(
                        _row(scene, speaker, arm, score,
                             n_accepted_before=None, system="no_recursion")
                    )

        lines = aggregate_phase3._table(rows, "real", "oracle", "n_accepted_before")
        rendered = "\n".join(lines)
        assert "no_recursion" not in rendered
        assert sum(_n_values(lines)) == 6
