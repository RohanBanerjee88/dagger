"""Offline Phase 3 fixtures: a corruptible fake diarizer and a scheduled scene.

Everything here runs with no pyannote, no GPU and no corpus — the same
constraint the rest of this suite works under (see ``tests/conftest.py``). The
point is that the whole ``run_phase3.py`` path, including cluster mapping and
the arm decomposition, is exercised offline; the real backend is a thin adapter
over these same contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dagger.data.activity import segments_from_chunks  # noqa: E402
from dagger.data.base import Scene  # noqa: E402
from dagger.data.mixing import mix_scheduled_sources, schedule_solo_then_overlap  # noqa: E402
from dagger.diarize.base import Diarizer  # noqa: E402
from dagger.diarize.oracle import Segment  # noqa: E402

SAMPLE_RATE = 8000


def make_tone(length: int, freq_hz: float, sample_rate: int = SAMPLE_RATE, amp: float = 0.5):
    t = np.arange(length, dtype=np.float64) / sample_rate
    return amp * np.sin(2.0 * np.pi * freq_hz * t)


def build_scheduled_scene(
    lengths: list[int],
    min_solo: int,
    sample_rate: int = SAMPLE_RATE,
    name: str = "scene",
) -> Scene:
    """A Phase 2-style scene: guaranteed per-speaker solo + a deep-overlap zone.

    Uses the real scheduler and mixer rather than hand-built masks, so the
    solo/overlap geometry these tests reason about is the same geometry the
    pipeline produces.
    """
    num = len(lengths)
    speakers = [f"s{i + 1}" for i in range(num)]
    raw = [make_tone(n, 220.0 * (i + 1), sample_rate) for i, n in enumerate(lengths)]
    chunks = schedule_solo_then_overlap(lengths, min_solo=min_solo)
    sources, mixture = mix_scheduled_sources(raw, chunks, length_mode="max")
    return Scene(
        mixture=mixture,
        sources=sources,
        segments=segments_from_chunks(chunks, speakers, sample_rate),
        speakers=speakers,
        sample_rate=sample_rate,
        name=name,
    )


class FakeDiarizer(Diarizer):
    """A deliberately imperfect diarizer, with each failure mode switchable.

    Real diarization does not fail in one way, and the whole point of Phase 3 is
    that different failures cost different amounts. Each knob below reproduces
    one, so a test can pin the consequence of that failure alone:

    * ``jitter`` — shifts every boundary by this many seconds (segmentation error)
    * ``drop_speaker`` — omits a speaker entirely (a missed speaker, k < m)
    * ``extra_cluster`` — invents a speaker from a slice of a real one (k > m)
    * ``relabel`` — emits anonymous ``SPEAKER_NN`` ids in a permuted order, which
      is what a real backend always does and what the cluster mapping must undo
    """

    binds_scene_speakers = False

    def __init__(
        self,
        *,
        jitter: float = 0.0,
        drop_speaker: int | None = None,
        extra_cluster: bool = False,
        relabel: bool = True,
        permutation: list[int] | None = None,
    ) -> None:
        self.jitter = jitter
        self.drop_speaker = drop_speaker
        self.extra_cluster = extra_cluster
        self.relabel = relabel
        self.permutation = permutation

    def diarize(self, scene) -> list[Segment]:
        speakers = list(scene.speakers)
        order = self.permutation or list(range(len(speakers)))[::-1]
        duration = scene.mixture.shape[0] / scene.sample_rate

        label_of: dict[str, str] = {}
        for rank, index in enumerate(order):
            label_of[speakers[index]] = (
                f"SPEAKER_{rank:02d}" if self.relabel else speakers[index]
            )

        out: list[Segment] = []
        for seg in scene.segments:
            if self.drop_speaker is not None and seg.speaker == speakers[self.drop_speaker]:
                continue
            start = max(0.0, seg.start + self.jitter)
            end = min(duration, seg.end + self.jitter)
            if end <= start:
                continue
            out.append(
                Segment(speaker=label_of[seg.speaker], start=start, duration=end - start)
            )

        if self.extra_cluster and out:
            longest = max(out, key=lambda s: s.duration)
            # Carve the middle third of a real segment into a phantom speaker.
            third = longest.duration / 3.0
            out.append(
                Segment(
                    speaker="SPEAKER_99",
                    start=longest.start + third,
                    duration=third,
                )
            )
        return out


@pytest.fixture
def three_speaker_scheduled_scene() -> Scene:
    """3 speakers, 1 s solo each, then a genuine depth-3 overlap zone."""
    return build_scheduled_scene(
        lengths=[SAMPLE_RATE * 4, SAMPLE_RATE * 4, SAMPLE_RATE * 4],
        min_solo=SAMPLE_RATE,
    )
