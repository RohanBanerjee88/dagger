"""Provenance tracking for the audio path (CLAUDE.md §1, guardrail §6.7).

THE ONE RULE: every speaker's output audio is extracted from the untouched
overlap mixture ``x_O`` — never from a residual (``x_O`` minus some estimate).
Extracting from a residual bakes each step's error into the next and makes the
total error grow with overlap depth; that is precisely the failure this project
exists to avoid.

This module makes the rule *mechanically enforceable*. A :class:`TrackedSignal`
carries a :class:`Provenance` tag describing how it was derived. Subtracting one
signal from another yields a ``RESIDUAL`` signal. The extractor ``G`` refuses to
run on a ``RESIDUAL`` input, raising :class:`ResidualInAudioPathError`. A unit
test (``tests/test_no_residual_in_audio_path.py``) asserts that the deflation
anti-pattern trips this guard while the correct pipeline does not.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import numpy as np


class Provenance(enum.Enum):
    """How a signal was derived, for the purpose of the audio-path rule."""

    #: The original, untouched mixture (or a masked slice of it, e.g. ``x_O``).
    #: Only signals with this provenance may feed the extractor ``G``.
    ORIGINAL_MIXTURE = "original_mixture"

    #: Derived by subtracting an estimate from another signal. Forbidden as an
    #: extractor input — this is the residual the accumulation-free proof rules
    #: out of the audio path.
    RESIDUAL = "residual"

    #: An extractor/estimate output, or anything else not covered above. Also
    #: not a valid extractor input (we never re-extract from an estimate).
    DERIVED = "derived"


class ResidualInAudioPathError(RuntimeError):
    """Raised when a residual is fed into the extractor to produce output audio.

    See CLAUDE.md §1. If you hit this, something is deflating a running residual
    into ``G`` — stop and re-read the audio-path rule.
    """


@dataclass
class TrackedSignal:
    """A waveform plus a provenance tag.

    Arithmetic propagates provenance so the audio-path rule stays checkable:

    * ``mixture - estimate`` -> :attr:`Provenance.RESIDUAL`
    * masking / scaling a mixture -> provenance preserved
    * anything else -> :attr:`Provenance.DERIVED`
    """

    samples: np.ndarray
    provenance: Provenance = Provenance.ORIGINAL_MIXTURE
    #: Free-form label for debugging, e.g. ``"x_O"`` or ``"x_O - s_hat_0"``.
    label: str = field(default="")

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=np.float64)

    @property
    def is_original_mixture(self) -> bool:
        return self.provenance is Provenance.ORIGINAL_MIXTURE

    def masked(self, mask: np.ndarray, *, label: str | None = None) -> "TrackedSignal":
        """Return ``samples * mask`` with provenance preserved.

        Masking an original mixture (e.g. to build ``x_O``) keeps it an original
        mixture — a masked slice of the mixture is still the mixture, not a
        residual.
        """
        return TrackedSignal(
            samples=self.samples * np.asarray(mask, dtype=np.float64),
            provenance=self.provenance,
            label=label if label is not None else self.label,
        )

    def __sub__(self, other: "TrackedSignal | np.ndarray") -> "TrackedSignal":
        other_samples = other.samples if isinstance(other, TrackedSignal) else np.asarray(other)
        other_label = other.label if isinstance(other, TrackedSignal) else "array"
        return TrackedSignal(
            samples=self.samples - other_samples,
            provenance=Provenance.RESIDUAL,
            label=f"({self.label} - {other_label})",
        )

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self.samples, dtype=dtype)

    def __len__(self) -> int:
        return len(self.samples)


def original_mixture(samples: np.ndarray, label: str = "x") -> TrackedSignal:
    """Wrap a raw waveform as an untouched original mixture."""
    return TrackedSignal(samples=samples, provenance=Provenance.ORIGINAL_MIXTURE, label=label)


def require_original_mixture(signal: TrackedSignal, *, context: str = "extractor G") -> np.ndarray:
    """Assert ``signal`` is an original mixture; return its samples.

    This is the choke point for the audio-path rule: every code path that feeds
    the extractor must route through here.
    """
    if not isinstance(signal, TrackedSignal):
        raise TypeError(
            f"{context} requires a TrackedSignal so provenance can be checked, "
            f"got {type(signal).__name__}."
        )
    if not signal.is_original_mixture:
        raise ResidualInAudioPathError(
            f"{context} was fed a {signal.provenance.value} signal "
            f"(label={signal.label!r}). The audio path must extract only from "
            f"the untouched overlap mixture x_O (CLAUDE.md §1)."
        )
    return signal.samples
