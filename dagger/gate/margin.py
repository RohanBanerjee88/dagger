"""The identity margin M_i (CLAUDE.md §2, §5 Phase 2).

"Leakage uses a MARGIN, not raw similarity": raw ``cos(s_hat_i, e_i)`` is
always positive (voices aren't orthogonal), so a bare similarity threshold
can't tell a correct extraction from a leaky one. The margin,
``M_i = cos(s_hat_i, e_i) - max_{j!=i} cos(s_hat_i, e_j)``, is what actually
distinguishes "sounds like speaker i" from "sounds a bit like everyone."

This gate is an *operational* part of the pipeline (it decides whether to
trust an estimate at inference time), not a reported metric -- so, unlike
``dagger.metrics.speaker_similarity`` (which must use a different encoder than
training to avoid metric-hygiene violations), this module embeds with the
*same* encoder phi used for enrollment: the margin has to live in the same
embedding space as e_i/e_bar_i to mean anything.
"""

from __future__ import annotations

import numpy as np

from dagger.enroll.encoder import SpeakerEncoder
from dagger.metrics.speaker_similarity import cosine_similarity


def identity_margin(
    estimate: np.ndarray,
    sample_rate: int,
    embedding_self: np.ndarray,
    embeddings_others: list[np.ndarray],
    encoder: SpeakerEncoder,
) -> float:
    """``M_i`` for one speaker's extracted estimate.

    Embeds ``estimate`` with ``encoder`` (phi), then returns
    ``cos(s_hat_i, e_i) - max_j cos(s_hat_i, e_j)``. ``nan`` if there are no
    other speakers to compare against (margin is undefined for a single-speaker
    scene).
    """
    if not embeddings_others:
        return float("nan")
    s_hat_i = encoder.embed(estimate, sample_rate)
    same = cosine_similarity(s_hat_i, embedding_self)
    other = max(cosine_similarity(s_hat_i, e_j) for e_j in embeddings_others)
    return same - other
