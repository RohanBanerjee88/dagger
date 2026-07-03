"""Evaluation metrics.

Phase 0 needs signal quality (SI-SDR). Later phases add the speaker identity
margin computed with a *different* encoder than ``phi`` (metric hygiene,
guardrail §6.3) and Whisper WER. Kept dependency-light for now: SI-SDR is pure
numpy.
"""

from dagger.metrics.sisdr import si_sdr, si_sdr_regionwise

__all__ = ["si_sdr", "si_sdr_regionwise"]
