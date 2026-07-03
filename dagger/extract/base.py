"""Extractor interface and the Phase 0 null extractor.

An :class:`Extractor` maps ``(x_O, embedding) -> waveform`` for one speaker. The
base class enforces the audio-path rule for every subclass: the input must be an
original-mixture :class:`~dagger.audio.provenance.TrackedSignal`, so no residual
can ever reach ``G`` (CLAUDE.md §1).
"""

from __future__ import annotations

import abc

import numpy as np

from dagger.audio.provenance import TrackedSignal, require_original_mixture


class Extractor(abc.ABC):
    """Base class for target-speaker extractors ``G(x_O, e_i)``."""

    def extract(self, x_O: TrackedSignal, embedding: np.ndarray) -> np.ndarray:
        """Extract the target speaker from the overlap mixture.

        ``x_O`` MUST be an original-mixture signal; this method raises
        :class:`~dagger.audio.provenance.ResidualInAudioPathError` otherwise.
        Subclasses implement :meth:`_extract` on the validated raw samples.
        """
        samples = require_original_mixture(x_O, context=f"{type(self).__name__}.extract")
        return self._extract(samples, np.asarray(embedding, dtype=np.float64))

    @abc.abstractmethod
    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        """Do the actual extraction on validated samples."""
        raise NotImplementedError


class NullExtractor(Extractor):
    """Phase 0 placeholder: returns silence (no extraction on overlaps).

    With this extractor the reconstruction copies solo regions and leaves the
    overlap contribution at zero, which is exactly the Phase 0 target: recover
    solo audio exactly, defer overlap extraction to Phase 1.
    """

    def _extract(self, x_O: np.ndarray, embedding: np.ndarray) -> np.ndarray:
        return np.zeros_like(x_O)
