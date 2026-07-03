"""dagger — diarization-guided, accumulation-free target speaker extraction.

The single rule that defines this project (CLAUDE.md §1): every speaker's
output waveform is computed from the *untouched* overlap mixture ``x_O`` —
never from a running residual. See :mod:`dagger.audio.provenance` for the
mechanism that makes a violation of that rule detectable in tests.
"""

__version__ = "0.0.0"
