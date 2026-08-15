"""The real diarizer: pyannote.audio 4.x ``community-1`` (CLAUDE.md §3).

Kept in its own module, and its heavy imports deferred into method bodies, so
``import dagger.diarize`` never requires the ``[diarize]`` extra. The extra is
deliberately NOT folded into ``[ml]``: CI installs ``dev,data`` plus CPU torch
only, and pyannote pulls a large tree plus a HuggingFace-gated checkpoint.

**Sample rate.** ``community-1`` is a 16 kHz model; this project's pipeline runs
at 8 kHz (CLAUDE.md §7). The seam is clean because :class:`Segment` boundaries
are in *seconds*: this class runs on whatever wideband audio the scene carries,
and ``activity_matrix`` converts seconds to samples at the pipeline's rate.

The preferred input is ``scene.mixture_hi`` — a mixture built natively at 16 kHz
from the same placement (see :mod:`dagger.data.librimix`). Upsampling the 8 kHz
mixture instead is supported but *warned about*, because it invents no content
above 4 kHz: the diarizer's DER would then be inflated for a reason that has
nothing to do with diarization difficulty, and that inflation would land straight
in the headline oracle-vs-real gap.
"""

from __future__ import annotations

import warnings

import numpy as np

from dagger.data.paths import get_credential
from dagger.diarize.base import DiarizationFailedError, Diarizer
from dagger.diarize.oracle import Segment

#: HF model id, per CLAUDE.md §3. Gated (free) — accept the terms on the model
#: page and put a token in DAGGER_HF_TOKEN.
DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"

#: What ``community-1`` expects.
MODEL_SAMPLE_RATE = 16000


def _to_segments(output) -> list[Segment]:
    """Convert whatever the pipeline returned into our :class:`Segment` list.

    pyannote 4.x returns a ``DiarizeOutput`` object, not the bare
    ``pyannote.core.Annotation`` that 3.x returned, so a fixed
    ``output.itertracks(...)`` raises ``AttributeError``. Rather than pin a
    version, unwrap defensively.

    **``speaker_diarization``, never ``exclusive_speaker_diarization``.** The
    4.x output carries both. The "exclusive" one assigns at most ONE speaker per
    instant — it exists to make reconciliation with transcript timestamps easy.
    Taking it here would silently delete every overlap from the diarization: the
    overlap mask would collapse toward empty, ``G`` would barely run, DER's
    ``overlap_recall`` would read ~0, and the whole experiment would be measuring
    a flattened diarization rather than a real one. That failure would produce
    numbers, not a crash, which is why the attribute is named explicitly and
    first.
    """
    annotation = getattr(output, "speaker_diarization", output)

    # pyannote.core.Annotation exposes itertracks; some versions' containers are
    # plain iterables of (segment, label) or (segment, track, label).
    if hasattr(annotation, "itertracks"):
        triples = annotation.itertracks(yield_label=True)
    else:
        triples = annotation

    segments: list[Segment] = []
    for item in triples:
        if len(item) == 3:
            turn, _track, label = item
        elif len(item) == 2:
            turn, label = item
        else:
            raise TypeError(
                f"unexpected diarization item {item!r} from pyannote; expected a "
                "(segment, label) or (segment, track, label) tuple."
            )
        duration = float(turn.end) - float(turn.start)
        if duration > 0:
            segments.append(
                Segment(speaker=str(label), start=float(turn.start), duration=duration)
            )
    return segments


class PyannoteDiarizer(Diarizer):
    """Wraps a pyannote ``Pipeline`` in the :class:`~dagger.diarize.base.Diarizer` contract.

    ``num_speakers`` is the diagnostic knob Phase 3 uses to separate two failure
    modes that otherwise arrive summed together:

    * ``None`` (default) — the diarizer estimates the speaker count itself. This
      is the honest "real" condition, and it *includes counting errors*: finding
      4 clusters where there are 3 is a genuine failure that costs real quality.
    * an ``int`` — the count is supplied. Miscounting is removed, so what remains
      is segmentation error alone.

    Reporting only the first leaves you unable to say which half of the gap you
    are looking at; reporting only the second hands the diarizer ground truth and
    understates the cost. Phase 3 runs both.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        num_speakers: int | None = None,
        token_env: str = "DAGGER_HF_TOKEN",
    ) -> None:
        self.model = model
        self.device = device
        self.num_speakers = num_speakers
        self.token_env = token_env
        self._pipeline = None  # built lazily; see _load()

    # -- pipeline construction -------------------------------------------------

    def _load(self):
        """Build the pipeline once, on first use.

        Lazy because constructing a ``PyannoteDiarizer`` happens at config-parse
        time (cheap, and done for every arm) while loading the checkpoint is
        expensive and needs credentials. Mirrors
        :class:`~dagger.enroll.encoder.TitaNetEncoder`'s fallback-chain idiom:
        an explicit local checkpoint path wins, otherwise fetch from the Hub.
        """
        if self._pipeline is not None:
            return self._pipeline

        import torch
        from pyannote.audio import Pipeline

        local = get_credential("DAGGER_PYANNOTE_CKPT_PATH")
        source = local or self.model
        token = get_credential(self.token_env)
        if local is None and token is None:
            warnings.warn(
                f"{self.token_env} is unset and DAGGER_PYANNOTE_CKPT_PATH is not "
                f"set either. {self.model!r} is a gated model; if the checkpoint "
                "is not already cached locally this will fail. Accept the terms "
                "on the model page and put a token in .env.",
                RuntimeWarning,
                stacklevel=2,
            )

        # pyannote renamed this kwarg between 3.x (`use_auth_token`) and 4.x
        # (`token`). Try the modern spelling first and fall back, rather than
        # pinning a version -- a TypeError here would surface as an opaque
        # failure at the top of an expensive run.
        try:
            pipeline = Pipeline.from_pretrained(source, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(source, use_auth_token=token)
        if pipeline is None:
            raise RuntimeError(
                f"pyannote returned no pipeline for {source!r}. This is what a "
                "rejected gated-model request looks like: accept the terms on "
                f"the model page and check {self.token_env}."
            )
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(torch.device(device))
        self._pipeline = pipeline
        return pipeline

    # -- audio selection -------------------------------------------------------

    def _waveform(self, scene) -> np.ndarray:
        """The wideband mixture to diarize, resampling only as a last resort."""
        hi = getattr(scene, "mixture_hi", None)
        hi_rate = getattr(scene, "hi_sample_rate", None)
        if hi is not None and hi_rate == MODEL_SAMPLE_RATE:
            return np.asarray(hi, dtype=np.float64)

        from dagger.data.audio_io import resample

        if hi is not None:
            return resample(np.asarray(hi), int(hi_rate), MODEL_SAMPLE_RATE)

        warnings.warn(
            "Scene has no mixture_hi; diarizing an UPSAMPLED "
            f"{scene.sample_rate} Hz mixture. Nothing above "
            f"{scene.sample_rate // 2} Hz exists in that signal, so DER will be "
            "inflated for reasons unrelated to diarization difficulty and that "
            "inflation lands in the oracle-vs-real gap. Set "
            "`dataset.diarizer_sample_rate: 16000` to build a native mixture.",
            RuntimeWarning,
            stacklevel=2,
        )
        return resample(
            np.asarray(scene.mixture), int(scene.sample_rate), MODEL_SAMPLE_RATE
        )

    # -- the contract ----------------------------------------------------------

    def diarize(self, scene) -> list[Segment]:
        import torch

        pipeline = self._load()
        waveform = self._waveform(scene)
        # pyannote wants [channel, time] float32.
        tensor = torch.as_tensor(waveform, dtype=torch.float32).unsqueeze(0)

        kwargs = {}
        if self.num_speakers is not None:
            kwargs["num_speakers"] = int(self.num_speakers)

        payload = {"waveform": tensor, "sample_rate": MODEL_SAMPLE_RATE}
        if self.num_speakers is None:
            # Free estimation: any failure here is a real problem, so let it
            # propagate rather than quietly shrinking the eval set.
            output = pipeline(payload)
        else:
            # A forced speaker count is UNSATISFIABLE on a short scene: if the
            # pipeline's segmentation yields fewer embedding windows than
            # `num_speakers`, its clustering raises (sklearn:
            # "n_samples=2 should be >= n_clusters=3"). That is a property of
            # the scene, not a bug, and it must not take down the whole run --
            # the forced-count arm is a diagnostic, while `real` and `oracle`
            # carry the headline. Wrapped so callers can skip THIS ARM on THIS
            # SCENE and keep everything else.
            try:
                output = pipeline(payload, **kwargs)
            except Exception as exc:  # noqa: BLE001 - narrowed by the branch above
                raise DiarizationFailedError(
                    f"pyannote could not diarize with num_speakers="
                    f"{self.num_speakers}: {type(exc).__name__}: {exc}"
                ) from exc
        return _to_segments(output)
