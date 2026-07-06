"""The speaker encoder phi: waveform -> fixed embedding vector.

phi is used only to build enrollment embeddings ``e_bar_i`` from solo clips
(CLAUDE.md guardrail: "eval encoder != training encoder" -- this module is the
*training*-side encoder; :mod:`dagger.metrics.speaker_similarity` holds the
separate eval-only encoder and never imports anything from this file).

phi is frozen throughout Phase 1 (CLAUDE.md §3: "Freeze pretrained weights
first; fine-tune late.") -- ``TitaNetEncoder.embed`` never takes a gradient.

Heavy imports (``torch``, ``nemo``) stay inside method bodies, not at module
import time, so ``import dagger.enroll`` does not require the ``[ml]`` extra
for callers who only need :mod:`dagger.enroll.topk`.
"""

from __future__ import annotations

import abc
import os
import tempfile
import warnings

import numpy as np

_TITANET_SAMPLE_RATE = 16000
_TITANET_MODEL_NAME = "nvidia/speakerverification_en_titanet_large"


class SpeakerEncoder(abc.ABC):
    """Maps a waveform clip to a fixed-length speaker embedding."""

    @abc.abstractmethod
    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a 1-D embedding for one waveform clip."""
        raise NotImplementedError


class TitaNetEncoder(SpeakerEncoder):
    """phi: NVIDIA NeMo's TitaNet-Large speaker-verification model.

    Apache-2.0 toolkit (``nemo_toolkit``), CC-BY-4.0 checkpoint (attributed in
    ``NOTICE``). Loaded once and reused across all enrollment calls; weights
    are frozen (``torch.no_grad()`` in :meth:`embed`).
    """

    def __init__(
        self,
        device: str = "cpu",
        checkpoint_source: str = "auto",
    ) -> None:
        """``checkpoint_source``: ``"auto"`` | ``"pretrained"`` | ``"local"`` | ``"random_init"``.

        ``"auto"`` tries a NGC/HuggingFace-backed ``from_pretrained`` fetch
        first, then a local ``.nemo`` file at ``DAGGER_TITANET_CKPT_PATH``, then
        falls back to a randomly-initialized model with a loud warning --
        shape/plumbing smoke tests only, never a reported research result.
        """
        self.device = device
        self._model = self._load_model(checkpoint_source)

    def _load_model(self, checkpoint_source: str):
        import torch

        def _from_pretrained():
            import nemo.collections.asr as nemo_asr

            return nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name=_TITANET_MODEL_NAME
            )

        def _from_local():
            import nemo.collections.asr as nemo_asr

            ckpt_path = os.environ.get("DAGGER_TITANET_CKPT_PATH")
            if not ckpt_path:
                raise FileNotFoundError(
                    "DAGGER_TITANET_CKPT_PATH is not set; no local TitaNet-Large "
                    "checkpoint to load."
                )
            return nemo_asr.models.EncDecSpeakerLabelModel.restore_from(ckpt_path)

        def _random_init():
            import nemo.collections.asr as nemo_asr

            warnings.warn(
                "TitaNetEncoder falling back to a randomly-initialized model -- "
                "embeddings are meaningless. Smoke-test / plumbing use only, "
                "never a valid research result.",
                stacklevel=2,
            )
            pretrained = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name=_TITANET_MODEL_NAME
            )
            pretrained.apply(
                lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
            )
            return pretrained

        loaders = {
            "pretrained": [_from_pretrained],
            "local": [_from_local],
            "random_init": [_random_init],
            "auto": [_from_pretrained, _from_local, _random_init],
        }
        if checkpoint_source not in loaders:
            raise ValueError(f"Unknown checkpoint_source {checkpoint_source!r}.")

        last_error: Exception | None = None
        for loader in loaders[checkpoint_source]:
            try:
                model = loader()
                break
            except Exception as exc:  # noqa: BLE001 -- deliberately broad fallback chain
                last_error = exc
                continue
        else:
            raise RuntimeError(
                f"Could not load TitaNet-Large via {checkpoint_source!r}."
            ) from last_error

        model = model.to(self.device)
        model.eval()
        return model

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return TitaNet-Large's speaker embedding for one waveform clip.

        NeMo's documented convenience API (``get_embedding``) takes a file
        path rather than a raw array, so the clip is written to a short-lived
        temp wav rather than reimplementing file I/O around NeMo's internal
        tensor path.
        """
        import soundfile as sf
        import torch

        from dagger.data.audio_io import resample

        clip = resample(np.asarray(waveform, dtype=np.float64), sample_rate, _TITANET_SAMPLE_RATE)

        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, clip.astype(np.float32), _TITANET_SAMPLE_RATE)
            with torch.no_grad():
                embedding = self._model.get_embedding(f.name)

        return np.asarray(embedding, dtype=np.float64).reshape(-1)
