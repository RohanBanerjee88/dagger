"""Evaluation metrics.

Signal quality (SI-SDR, ``dagger.metrics.sisdr``, pure numpy) plus, from
Phase 1, the speaker identity margin (``dagger.metrics.speaker_similarity``)
computed with a *different* encoder than ``phi`` (metric hygiene, guardrail
§6.3). Whisper WER remains a later phase.
"""

from dagger.metrics.sisdr import si_sdr, si_sdr_best_permutation, si_sdr_regionwise

__all__ = ["si_sdr", "si_sdr_regionwise", "si_sdr_best_permutation"]
