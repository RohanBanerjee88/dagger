"""The oracle-refinement CEILING: an acceptance rule that can see the answer.

**NOT DEPLOYABLE. Scoring-time only.** This module reads ``scene.sources`` --
the clean per-speaker ground truth -- to decide whether a refinement candidate
is kept. Nothing here may ever run in a reported system; it exists to put an
upper bound on a component we are about to report as net-harmful.

Why a ceiling is needed
-----------------------
Confidence-gated refinement (:mod:`dagger.refine.coarse_to_fine`) has measured
net-negative in every regime tested so far:

* clean enrollment (Phase 2)                    -0.07 to -0.54 dB
* heterogeneous enrollment (Phase 2)            -0.36 / -0.69 dB
* contaminated real-diarization (Phase 3 A)     -0.49 dB

That is three independent negatives, and it still does not establish "never
positive" -- accumulating negatives never can. Worse, the mechanism predicts the
pattern without settling the question: refinement blends
``0.5*e_enrolled + 0.5*e_from_extracted_overlap``, and the second term is
embedded from ``G``'s output, which sits at ~1.7 dB at depth 2. Bad conditioning
produces a bad candidate, so the two requirements fight, and every test lands
negative for a reason that might be the *extractor's* fault rather than
refinement's.

Replacing the acceptance rule with one that cannot be wrong separates them, and
whichever way it lands is a result:

* **ceiling still negative** -- refinement has no headroom on this extractor.
  A publishable negative result with a stated mechanism, not an open question.
* **ceiling positive, real gate cannot find it** -- the acceptance RULE is what
  is broken, not refinement. The levers then are the ones Phase 2's close-out
  already flagged: a variance-weighted blend instead of the fixed 0.5/0.5, and
  embedding the lowest-depth frames rather than the longest overlap run.

Why the rule scores AUDIO, not embeddings
-----------------------------------------
The obvious formulation -- accept iff the candidate embedding is closer to the
true speaker's embedding -- does not typecheck, and the way it fails is
instructive. Candidate and current embeddings live in ``phi``'s TitaNet space;
an eval-encoder reference lives in WavLM space. The two are not comparable, and
the only way to make them so is to embed the true source with ``phi`` -- exactly
the training-encoder-as-metric violation CLAUDE.md §6.3 forbids.

Scoring the reconstructed waveform against the clean source sidesteps this
entirely: no encoder is involved, so §6.3 cannot be violated.

Why the rule scores the SAME SLICE the metric reports
-----------------------------------------------------
**This is the correctness property that makes the number a ceiling at all, and
the first version got it wrong.** The 2026-08-19 run scored candidates on the
WHOLE waveform while the results table reported ``si_sdr_by_depth`` -- depth 2
only. Different objectives, so the monotonicity argument did not transfer, and
the "ceiling" came in 0.24-0.37 dB BELOW ``no_recursion``: a bound that cannot
lose, losing.

The mechanism is SI-SDR's scale invariance. It fits a scalar ``alpha`` to the
estimate before measuring the residual, and *which samples you include decides
what alpha becomes*. Over the whole waveform, ~75% of the audio is a bit-exact
solo copy, which pins ``alpha`` near 1 and makes any level error in the overlap
region count in full. Over the depth-2 slice alone, ``alpha`` floats freely and
absorbs that level error for nothing. So a candidate that fixes ``G``'s *level*
while worsening its *shape* wins the whole-waveform comparison and loses the one
that gets reported -- which is exactly what happened, on 27/75 and 31/75 speakers
respectively, at a mean cost of 0.67-0.89 dB.

Hence :func:`make_oracle_accept_fn` takes ``scoring_depth`` and masks with it,
via the same :func:`~dagger.metrics.sisdr.si_sdr_regionwise` that
:func:`~dagger.metrics.sisdr.si_sdr_by_depth` uses. The guarantee is then real
and mechanically checkable: **the optimized quantity cannot decrease**, so a
still-negative result is a genuine ceiling rather than a mis-specified
objective. ``tests/phase3/test_refine_oracle_ceiling.py`` asserts it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from dagger.metrics.sisdr import si_sdr_regionwise

#: Lowest overlap depth the ceiling optimizes. Refinement can only change frames
#: the extractor produces (``w_Oi > 0``), which are the overlap frames; depth 1
#: is copied straight from the mixture and is not the embedding's to improve.
DEFAULT_MIN_DEPTH = 2


def make_oracle_accept_fn(
    row_sources: Sequence[np.ndarray | None],
    scoring_depth: np.ndarray,
    min_depth: int = DEFAULT_MIN_DEPTH,
) -> Callable[[int, np.ndarray, np.ndarray], bool]:
    """An ``accept_fn`` for :func:`~dagger.refine.coarse_to_fine.refine_embeddings`.

    Accepts a refinement candidate iff the audio reconstructed from it scores
    strictly higher SI-SDR against the true source **over the frames at depth
    >= min_depth**. Ties reject, so the ceiling never counts a no-op as a win --
    with a strict inequality the measured ceiling is a lower bound on the
    achievable one, the safe direction for a bound whose *negative* reading is
    the publishable outcome.

    ``row_sources[i]`` is the clean ground-truth source for **refiner row** ``i``,
    or ``None`` when that row has no ground-truth counterpart. Rows, not sources:
    under real diarization ``score_scene`` drops unenrollable clusters and then
    re-indexes every per-speaker array, so refiner row ``i`` is generally not
    source ``i``. Taking the already-restricted, already-attributed per-row
    sources (``score_scene``'s ``targets``) removes that mapping from the
    caller's hands entirely -- a wrong map would score each speaker against
    another speaker's audio and report a confidently wrong ceiling, with nothing
    failing.

    ``scoring_depth`` is the per-sample reference depth ``|K|(t)`` -- the SAME
    array ``score_scene`` passes to :func:`~dagger.metrics.sisdr.si_sdr_by_depth`
    to build the results table. Masking with it, through the same
    :func:`~dagger.metrics.sisdr.si_sdr_regionwise`, is what makes the ceiling
    a ceiling; see this module's docstring for what went wrong when it scored
    the whole waveform instead.

    **The exact guarantee, and its one limit.** What cannot decrease is SI-SDR
    *pooled over all depths >= min_depth*. When the corpus has a single overlap
    depth -- as the Phase 3 long-scene chain corpus does, where depth stops at 2 --
    that pooled quantity IS the reported per-depth number, and the guarantee is
    exactly the one the table needs. With several overlap depths present, an
    individual depth may still move either way while the pool improves. The
    strict alternative (require improvement at every depth separately) is more
    conservative and would understate the ceiling, so it is deliberately not the
    default; reach for it only if a per-depth guarantee is what a claim rests on.

    A ``None`` row rejects: a spurious cluster has no true speaker to get closer
    to, so no candidate for it can be an improvement.
    """
    mask = np.asarray(scoring_depth) >= int(min_depth)

    def accept(i: int, candidate_output: np.ndarray, current_output: np.ndarray) -> bool:
        target = row_sources[i]
        if target is None:
            return False
        improved = si_sdr_regionwise(candidate_output, target, mask)
        baseline = si_sdr_regionwise(current_output, target, mask)
        if np.isnan(improved) or np.isnan(baseline):
            # Either no frames at this depth, or no target energy in them -- so
            # "better" is undefined. Reject: an undefined comparison must not be
            # scored as an improvement. (`si_sdr_regionwise` returns nan for an
            # empty mask, which covers the no-overlap-in-this-scene case too.)
            return False
        return bool(improved > baseline)

    return accept
