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
entirely: no encoder is involved, so §6.3 cannot be violated, and the rule
optimizes the quantity actually reported (SI-SDR) rather than a proxy for it.
That makes this the *tighter* ceiling as well as the cleaner one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from dagger.metrics.sisdr import si_sdr


def make_oracle_accept_fn(
    row_sources: Sequence[np.ndarray | None],
) -> Callable[[int, np.ndarray, np.ndarray], bool]:
    """An ``accept_fn`` for :func:`~dagger.refine.coarse_to_fine.refine_embeddings`.

    Accepts a refinement candidate iff the audio reconstructed from it scores
    strictly higher SI-SDR against the true source than the audio reconstructed
    from the current embedding. Ties reject, so the ceiling never counts a
    no-op as a win -- with a strict inequality the measured ceiling is a lower
    bound on the achievable one, which is the safe direction for a bound whose
    *negative* reading is the publishable outcome.

    ``row_sources[i]`` is the clean ground-truth source for **refiner row** ``i``,
    or ``None`` when that row has no ground-truth counterpart. Rows, not sources:
    under real diarization ``score_scene`` drops unenrollable clusters and then
    re-indexes every per-speaker array, so refiner row ``i`` is generally not
    source ``i``. Taking the already-restricted, already-attributed per-row
    sources (``score_scene``'s ``targets``) removes that mapping from the
    caller's hands entirely -- a wrong map would score each speaker against
    another speaker's audio and report a confidently wrong ceiling, with nothing
    failing.

    A ``None`` row rejects: a spurious cluster has no true speaker to get closer
    to, so no candidate for it can be an improvement.
    """
    def accept(i: int, candidate_output: np.ndarray, current_output: np.ndarray) -> bool:
        target = row_sources[i]
        if target is None:
            return False
        improved = si_sdr(candidate_output, target)
        baseline = si_sdr(current_output, target)
        if np.isnan(improved) or np.isnan(baseline):
            # No target energy here, so "better" is undefined. Reject: an
            # undefined comparison must not be scored as an improvement.
            return False
        return bool(improved > baseline)

    return accept
