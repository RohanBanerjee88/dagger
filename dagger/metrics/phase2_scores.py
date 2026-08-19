"""Shared reading/aggregation layer for Phase 2's score CSVs.

Three consumers -- ``scripts/run_phase2.py`` (writes them), and
``scripts/aggregate_phase2.py`` / ``scripts/plot_phase2_depth.py`` (read them) --
used to carry three independent copies of the clipping rule, each with a comment
saying it "matches run_phase2.py's". That duplication has already cost us once:
the 2026-07-26 ``+-inf`` bug had to be found and fixed in two scripts separately,
and a plot that silently disagrees with the table beside it is exactly the class
of defect Phase 2 kept shipping. One definition, three importers.

The two rules that must never drift:

* ``nan`` means "this speaker is not active at this depth, there is nothing to
  score" -- drop it.
* ``+-inf`` means "a perfect estimate" / "a totally failed estimate against real
  target energy" (see ``dagger.metrics.sisdr.si_sdr``). Those are *informative
  outcomes*, not undefined ones -- clip them to ``+-SI_SDR_CAP_DB`` so a handful
  of perfect solo-copy rows can neither vanish from a mean nor make it infinite.
  A plain ``np.isfinite()`` filter drops both alongside ``nan``; that was the
  bug, and it understated depth 1 by 7-8 dB.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Sequence
from pathlib import Path

SYSTEMS = ("no_recursion", "ungated_deflation", "gated_deflation", "coarse_to_fine")
DEFLATION_SYSTEMS = ("ungated_deflation", "gated_deflation")
CONTROL_SYSTEM = "no_recursion"

SI_SDR_CAP_DB = 50.0

#: Columns carrying an integer that is legitimately blank for the
#: accumulation-free systems (0 would mean "nothing deflated before me", which is
#: a different statement from "this system never deflates").
_OPTIONAL_INT_FIELDS = ("deflation_index", "n_accepted_before", "refine_rounds")
_REQUIRED_INT_FIELDS = ("m", "depth")


def clip_score(value: float) -> float | None:
    """``None`` for ``nan`` (exclude it); otherwise clipped to ``+-SI_SDR_CAP_DB``."""
    if math.isnan(value):
        return None
    return max(-SI_SDR_CAP_DB, min(SI_SDR_CAP_DB, float(value)))


def scoreable(values: Iterable[float]) -> list[float]:
    """Apply :func:`clip_score` across an iterable, dropping the ``nan`` rows."""
    return [c for c in (clip_score(v) for v in values) if c is not None]


def load_score_rows(
    csv_paths: Sequence[Path], *, required_columns: Sequence[str] = ("m",)
) -> list[dict]:
    """Concatenate score CSVs into typed dicts, keeping only scoreable rows.

    A file missing any of ``required_columns`` is rejected loudly rather than
    silently folded in -- an older CSV predating the accumulation instrumentation
    would otherwise aggregate into a single meaningless group, which looks like a
    result rather than a mistake.
    """
    rows: list[dict] = []
    for path in csv_paths:
        path = Path(path)
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            missing = [c for c in required_columns if c not in (reader.fieldnames or ())]
            if missing:
                raise SystemExit(
                    f"{path} has no {', '.join(repr(c) for c in missing)} column -- it "
                    "predates the accumulation instrumentation. Re-run "
                    "scripts/run_phase2.py to regenerate it."
                )
            for row in reader:
                score = clip_score(float(row["si_sdr"]))
                if score is None:  # speaker not active here, nothing to score
                    continue
                record = {
                    "source": path.name,
                    "scene": row.get("scene"),
                    "speaker": row.get("speaker"),
                    "system": row["system"],
                    "si_sdr": score,
                    # Phase 3's arm name (oracle / real / ...). Absent in every
                    # Phase 2 CSV, where it reads as None -- so those files keep
                    # loading unchanged and a mixed set stays distinguishable.
                    "diarization": row.get("diarization"),
                    # Phase 3's overlap-dilation sweep point, in ms. Absent
                    # (Phase 2, and Phase 3 runs predating the sweep) reads as
                    # 0.0 = undilated, which is what those runs were. Carried
                    # so downstream pairing can hold it FIXED: a swept CSV
                    # contains several different pipelines, and averaging across
                    # them would relabel a mean of pipelines as a property of
                    # one -- the same two-variables-in-one-column mistake that
                    # cost Phase 2 five runs on the depth axis.
                    "dilate_ms": float(row.get("dilate_ms") or 0.0),
                }
                for field in _REQUIRED_INT_FIELDS:
                    record[field] = int(row[field])
                for field in _OPTIONAL_INT_FIELDS:
                    raw = row.get(field)
                    record[field] = int(raw) if raw not in (None, "") else None
                rows.append(record)
    if not rows:
        raise SystemExit("no scoreable rows found in the given CSVs")
    return rows


def mean_sem(values: Sequence[float]) -> tuple[float, float, int]:
    """``(mean, standard error of the mean, n)``.

    SEM is ``sd/sqrt(n)`` -- the precision of the *average*, which shrinks as
    ``n`` grows. It is NOT the spread of the data (that is ``sd``, or the p5-p95
    band), and on this project's numbers the two differ by roughly 30x. See
    CLAUDE.md §7 "Statistical reporting" for which to report when. ``nan`` SEM
    for ``n < 2``, where it is undefined.
    """
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = math.fsum(values) / n
    if n < 2:
        return mean, float("nan"), n
    variance = math.fsum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance) / math.sqrt(n), n


def group_values(
    rows: Iterable[dict],
    x_field: str,
    *,
    depth: int | None = None,
    systems: Sequence[str] = SYSTEMS,
) -> dict[str, dict[int, list[float]]]:
    """``{system: {x: [si_sdr, ...]}}``, optionally at one fixed overlap depth.

    Rows whose ``x_field`` is ``None`` are skipped -- see ``_OPTIONAL_INT_FIELDS``
    on why blank is not zero.
    """
    grouped: dict[str, dict[int, list[float]]] = {}
    for row in rows:
        if row["system"] not in systems:
            continue
        if depth is not None and row["depth"] != depth:
            continue
        x = row.get(x_field)
        if x is None:
            continue
        grouped.setdefault(row["system"], {}).setdefault(int(x), []).append(row["si_sdr"])
    return grouped


def terminal_x_values(rows: Iterable[dict], x_field: str) -> dict[str, set[int]]:
    """``{system: {x, ...}}`` for x-positions that are a scene's LAST deflation step.

    The final step of a deflation chain is the **one-and-rest endpoint**: its
    residual is the mixture minus every other estimate, which already
    approximates that speaker's own source, so the extractor has an unusually
    easy job and the curve flattens or ticks up there. It is a benign special
    case of Theorem 2's ``||E_m|| <= m*eps`` upper bound, not a trend point, and
    plotting it as one invites a reader to conclude the theory failed.

    Only meaningful for ``x_field='n_accepted_before'``, where terminal means
    ``x == m - 1``. A position counts only if *every* contributing row is
    terminal, so a mixed bucket is left as an ordinary point.
    """
    if x_field != "n_accepted_before":
        return {}
    tally: dict[str, dict[int, list[bool]]] = {}
    for row in rows:
        x = row.get(x_field)
        if x is None:
            continue
        tally.setdefault(row["system"], {}).setdefault(int(x), []).append(
            int(x) == row["m"] - 1
        )
    return {
        system: {x for x, flags in per_x.items() if all(flags)}
        for system, per_x in tally.items()
    }


def control_slope(
    rows: Iterable[dict], x_field: str, *, depth: int | None = None
) -> tuple[float, int, int] | None:
    """``(slope_db, x_min, x_max)`` for the ``no_recursion`` control, or ``None``.

    ``no_recursion`` takes zero deflation steps, so any dependence it shows on an
    accumulation axis is eval-set difficulty rather than accumulation. A raw
    cross-eval-set sweep is only readable while this is ~flat. It was +0.16 dB on
    the 2026-08-04 checkpoint -- which is why depth 2 was chosen for that figure --
    and -0.95 dB on the 2026-08-09 one, at which point the same figure silently
    became four near-parallel declining lines. Checking it by hand once and
    treating it as settled is what failed; callers should check it every run.
    """
    grouped = group_values(rows, x_field, depth=depth, systems=(CONTROL_SYSTEM,))
    per_x = grouped.get(CONTROL_SYSTEM)
    if not per_x or len(per_x) < 2:
        return None
    xs = sorted(per_x)
    first, _, _ = mean_sem(per_x[xs[0]])
    last, _, _ = mean_sem(per_x[xs[-1]])
    return last - first, xs[0], xs[-1]


def paired_differences(
    rows: Iterable[dict],
    system: str,
    control: str = CONTROL_SYSTEM,
    *,
    key_fields: Sequence[str] = ("source", "scene", "speaker", "depth"),
) -> list[float]:
    """Per-row ``system - control`` differences over rows both systems scored.

    Pairing on the same ``(scene, speaker, depth)`` cancels scene difficulty
    *exactly*, rather than hoping it averages out across two different corpora.
    That is what lets an ``m`` sweep span several eval sets without the
    control needing to be flat: the control becomes 0 by construction.
    """
    def key(row: dict) -> tuple:
        return tuple(row[f] for f in key_fields)

    left = {key(r): r["si_sdr"] for r in rows if r["system"] == system}
    right = {key(r): r["si_sdr"] for r in rows if r["system"] == control}
    return [left[k] - right[k] for k in left if k in right]


def paired_rows_by_field(
    rows: Iterable[dict],
    field: str,
    left_value: str,
    right_value: str,
    *,
    key_fields: Sequence[str] = ("source", "scene", "speaker", "depth", "system"),
) -> list[tuple[dict, dict]]:
    """``(left_row, right_row)`` for every row both values of ``field`` scored.

    :func:`paired_by_field` throws the rows away and keeps only the differences,
    which is all a ``depth`` stratification needs -- the two arms always agree on
    depth, because it is derived from the REFERENCE activity for both (see
    ``dagger.eval.systems.score_scene``). It is NOT enough on the accumulation
    axis, where the two arms routinely disagree: deflation order is ascending
    ``V_i``, identically 0 under oracle diarization but a real permutation under
    a real diarizer, so a speaker's ``n_accepted_before`` is arm-dependent by
    construction.

    A caller that filters on such a field *before* pairing therefore keeps only
    the speakers whose position happened to coincide across arms -- which is
    exactly the quantity under study, so the surviving sample is biased rather
    than merely small. On the Stage A run that showed up as n = 34/28/16 against
    n = 98/92/92 for the arm whose order matches oracle by construction.

    Returning the rows lets the caller pair first and bucket afterwards, on
    whichever arm is the reference. See ``scripts/aggregate_phase3.py::_table``.
    """
    def key(row: dict) -> tuple:
        return tuple(row[f] for f in key_fields)

    left = {key(r): r for r in rows if r.get(field) == left_value}
    right = {key(r): r for r in rows if r.get(field) == right_value}
    return [(left[k], right[k]) for k in left if k in right]


def paired_by_field(
    rows: Iterable[dict],
    field: str,
    left_value: str,
    right_value: str,
    *,
    key_fields: Sequence[str] = ("source", "scene", "speaker", "depth", "system"),
) -> list[float]:
    """Per-row ``left - right`` differences across two values of ``field``.

    The general form of :func:`paired_differences`, which pairs across *systems*
    at fixed conditions; this pairs across any other column at fixed system.
    Phase 3 uses it on ``diarization`` to difference the oracle and real arms,
    which is why ``system`` moves into the key: the comparison is "the same
    system under two diarizers", not "two systems".

    Same reason for pairing rather than subtracting two means -- it cancels scene
    difficulty *exactly*. Here that matters even more than in Phase 2, since both
    arms score the identical scenes in the identical pass, so every non-arm
    source of variance is shared and cancels.
    """
    def key(row: dict) -> tuple:
        return tuple(row[f] for f in key_fields)

    left = {key(r): r["si_sdr"] for r in rows if r.get(field) == left_value}
    right = {key(r): r["si_sdr"] for r in rows if r.get(field) == right_value}
    return [left[k] - right[k] for k in left if k in right]
