"""Deflation reconstruction -- the DELIBERATE anti-pattern (CLAUDE.md §1, §5 Phase 2).

*** WARNING: this module intentionally implements the audio-path violation ***
*** CLAUDE.md §1 forbids everywhere else in the pipeline. It exists ONLY as   ***
*** the ungated/gated deflation comparison baselines for Phase 2's depth-    ***
*** stratified experiment -- proof that subtracting a running estimate from  ***
*** the mixture and re-extracting from that residual accumulates error with  ***
*** overlap depth, unlike the accumulation-free path                        ***
*** (:mod:`dagger.reconstruct.stitch`) that :mod:`dagger.refine.coarse_to_fine`***
*** and Phase 1's "no_recursion" system use.                                 ***

Do not import this module from :mod:`dagger.refine` or any code that produces
the pipeline's real output audio. ``dagger.extract.base.Extractor.extract()``
raises :class:`~dagger.audio.provenance.ResidualInAudioPathError` if handed a
residual precisely to make this anti-pattern hard to do by accident;
:func:`_extract_from_residual` is the one deliberate, explicit bypass.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from dagger.audio.provenance import Provenance, TrackedSignal
from dagger.extract.base import Extractor
from dagger.gate.confidence import GateResult
from dagger.reconstruct.stitch import crossfade_windows


def _extract_from_residual(extractor: Extractor, residual: TrackedSignal, embedding: np.ndarray) -> np.ndarray:
    """THE ONLY call site in the repo allowed to feed a residual into ``G``.

    Bypasses :meth:`Extractor.extract`'s guard on purpose by calling
    ``extractor._extract`` directly on the residual's raw samples. Never call
    this outside :mod:`dagger.reconstruct.deflation`.
    """
    return extractor._extract(
        np.asarray(residual.samples, dtype=np.float64), np.asarray(embedding, dtype=np.float64)
    )


def reconstruct_all_deflation(
    x: TrackedSignal,
    x_O: TrackedSignal,
    activity: np.ndarray,
    solo: np.ndarray,
    embeddings: np.ndarray,
    extractor: Extractor,
    order: list[int],
    *,
    gate_fn: Optional[Callable[[int, np.ndarray], GateResult]] = None,
    fade: int = 0,
) -> tuple[np.ndarray, list[Optional[GateResult]]]:
    """Iterative residual-deflation reconstruction, for comparison only.

    Speakers are processed in ``order`` (the caller's choice -- e.g. ascending
    enrollment variance, most-confident first). The first speaker in ``order``
    extracts from ``x_O`` itself (no prior estimate exists yet); every
    subsequent speaker extracts from ``x_O`` minus the *accepted* prior
    estimates' overlap contributions.

    * ``gate_fn=None`` -- ungated_deflation: every estimate unconditionally
      updates the running residual (the naive anti-pattern).
    * ``gate_fn=callable`` -- gated_deflation: ``gate_fn(speaker_idx, estimate)``
      returns a :class:`~dagger.gate.confidence.GateResult`; a rejected
      estimate is still returned as that speaker's own output, it simply
      leaves the running residual untouched for the next speaker (so it can't
      contaminate anyone else's extraction).

    Returns ``(outputs [S, T], gate_results)``; ``gate_results[i]`` is ``None``
    when ``gate_fn`` is ``None``, else that speaker's :class:`GateResult`.
    """
    num_speakers = activity.shape[0]
    outputs = np.zeros_like(activity, dtype=np.float64)
    gate_results: list[Optional[GateResult]] = [None] * num_speakers

    x_samples = np.asarray(x, dtype=np.float64)
    running_residual = x_O
    for i in order:
        w_Ei, w_Oi = crossfade_windows(solo[i], activity[i], fade=fade)
        g_out = _extract_from_residual(extractor, running_residual, embeddings[i])
        outputs[i] = x_samples * w_Ei + np.asarray(g_out, dtype=np.float64) * w_Oi

        accept = True
        if gate_fn is not None:
            result = gate_fn(i, outputs[i])
            gate_results[i] = result
            accept = result.accepted

        if accept:
            overlap_contribution = TrackedSignal(
                samples=np.asarray(g_out, dtype=np.float64) * w_Oi,
                provenance=Provenance.DERIVED,
                label=f"g_out_{i}",
            )
            running_residual = running_residual - overlap_contribution

    return outputs, gate_results
