"""Eval-only speaker-similarity/margin diagnostic (CLAUDE.md §2/§6.3).

"The gate can't check its own enrollment" and "eval encoder != training
encoder" are non-negotiable per CLAUDE.md: computing speaker similarity with
the same encoder used to build ``e_bar_i`` (:mod:`dagger.enroll.encoder`) is a
metric-hygiene violation, one of Phase 1's named red flags. This module
deliberately never imports ``dagger.enroll.encoder`` -- it wraps a completely
different model (WavLM via ``transformers``, not NeMo TitaNet), so the eval
encoder physically cannot alias phi.

This computes the Phase 1 red-flag diagnostic (the margin ``M_i``) as a
reported column, not the full Phase 2 confidence gate -- there is no
accept/reject threshold ``tau`` here, just a number in the results table.
"""

from __future__ import annotations

import numpy as np

from dagger.enroll.topk import mean_embedding, select_topk_solo_clips

_WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus-sv"
_WAVLM_SAMPLE_RATE = 16000


class EvalSpeakerEncoder:
    """The eval-only encoder: ``microsoft/wavlm-base-plus-sv`` via ``transformers``.

    Architecturally unrelated to :class:`dagger.enroll.encoder.TitaNetEncoder`
    (WavLM vs. NeMo TitaNet), so there is no risk of the eval encoder
    accidentally sharing weights or lineage with phi.
    """

    def __init__(self, device: str = "cpu") -> None:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

        self.device = device
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(_WAVLM_MODEL_NAME)
        self._model = WavLMForXVector.from_pretrained(_WAVLM_MODEL_NAME).to(device)
        self._model.eval()

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        import torch

        from dagger.data.audio_io import resample

        clip = resample(np.asarray(waveform, dtype=np.float64), sample_rate, _WAVLM_SAMPLE_RATE)
        inputs = self._feature_extractor(
            [clip.astype(np.float32)], sampling_rate=_WAVLM_SAMPLE_RATE, return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            embedding = self._model(**inputs).embeddings[0]
        return embedding.cpu().numpy().astype(np.float64)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def eval_enroll_and_margin(
    mixture: np.ndarray,
    solo: np.ndarray,
    activity: np.ndarray,
    outputs: np.ndarray,
    sample_rate: int,
    encoder: EvalSpeakerEncoder,
    k: int = 3,
) -> list[float]:
    """Per-speaker identity margin ``M_i`` of the reconstructed outputs.

    Re-derives its *own* mean eval-space embeddings ``e_i`` per speaker from
    the same solo clips (:func:`dagger.enroll.topk.select_topk_solo_clips`,
    which is encoder-agnostic and safe to share here -- only
    :class:`~dagger.enroll.encoder.TitaNetEncoder` is never imported by this
    module), then computes the margin -- never raw similarity --
    ``M_i = cos(s_hat_i, e_i) - max_{j!=i} cos(s_hat_i, e_j)``, all in
    WavLM-embedding space, never mixing it with phi's TitaNet-space vectors.
    """
    num_speakers = outputs.shape[0]
    eval_embeddings = []
    for i in range(num_speakers):
        clips = select_topk_solo_clips(mixture, solo[i], sample_rate, k=k, activity_i=activity[i])
        e_i, _ = mean_embedding(clips, sample_rate, encoder)
        eval_embeddings.append(e_i)

    margins = []
    for i in range(num_speakers):
        s_hat_i = encoder.embed(outputs[i], sample_rate)
        same = cosine_similarity(s_hat_i, eval_embeddings[i])
        others = [
            cosine_similarity(s_hat_i, eval_embeddings[j])
            for j in range(num_speakers)
            if j != i
        ]
        margins.append(same - max(others) if others else float("nan"))
    return margins
