#!/usr/bin/env python3
"""Build LONG scenes (minutes, not seconds) so a real diarizer can actually work.

Why this exists
---------------
Phase 3 Stage A kept collapsing: pyannote returned ~2 clusters for 3-speaker
scenes in EVERY geometry tried -- scheduled/1 s solo (149/150 scenes degenerate),
chain (70/150), scheduled/3 s solo (42/150). Cluster count was 2.04 / 2.07
regardless, so geometry changed only whether the second cluster was enrollable,
never how many speakers were found.

The remaining suspect is scene LENGTH. Stock LibriMix scenes here are 10-20 s.
pyannote slides a segmentation window over the audio, embeds each local speaker,
then clusters those embeddings globally -- on 20 s that is a handful of
embeddings, most of them contaminated by overlap. Standard diarization
benchmarks are minutes: CALLHOME conversations run 2-5 minutes, AMI meetings
tens of minutes. We have been asking the diarizer to work an order of magnitude
below its design point.

What this generates
-------------------
Rows whose sources are SEVERAL utterances of the same speaker concatenated,
written ``p1|p2|p3|...`` -- a shape :class:`dagger.data.librimix.LibriMixDataset`
already supports (it is how the heterogeneous corpus works). No loader changes.

The knob is ``--per-speaker-sec``, the audio each speaker contributes. Scene
length follows from placement, and for ``chain`` with n speakers at overlap
``o``:

    scene_sec ~= (1 + (n - 1) * (1 - o)) * per_speaker_sec

so 3 speakers at overlap 0.3 gives ``2.4 * per_speaker_sec`` -- 50 s each for a
2-minute scene. The predicted length is printed; check it matches your intent.

Do NOT use overlap 0.5 with 3 speakers
--------------------------------------
Chain placement puts the middle speaker between the other two, and at
``overlap = 0.5`` its solo window closes **exactly**: s1 ends at the same instant
s3 begins. Measured on a 129 s scene, solo per speaker came out
``32 s / 1 s / 33 s`` -- the middle speaker keeping only the ``min_solo``
floor. That is the same starvation that made pyannote return 2 clusters in the
first place, so a long-scene run at overlap 0.5 would reproduce the failure it
was built to escape. The general condition is ``overlap < 0.5``; measured
alternatives at 60 s per speaker:

    overlap 0.5 -> solo 32/ 1/33 s, 49% overlapped   <- middle speaker starved
    overlap 0.4 -> solo 38/13/38 s, 36% overlapped
    overlap 0.3 -> solo 45/26/45 s, 25% overlapped   <- recommended
    overlap 0.2 -> solo 51/38/51 s, 15% overlapped   (closest to AMI)

Cost warning
------------
Runtime scales with audio duration. 2-minute scenes are ~6x the stock ones, so
150 of them is roughly a 6x eval. Use ~50 scenes for a comparable wall time;
50 scenes x 3 speakers x 3 depths still yields ~450 scored rows.

    python scripts/build_long_scene_metadata.py \\
        --librispeech-root /kaggle/working/data/LibriSpeech/test-clean \\
        --output /kaggle/working/data/metadata/Libri3Mix/libri3mix_test_long.csv \\
        --n-src 3 --num-scenes 50 --per-speaker-sec 60
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

SAMPLE_RATE_HINT = 16000  # LibriSpeech native; only used if soundfile can't report


def _utterance_duration(path: Path) -> float:
    import soundfile as sf

    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate or SAMPLE_RATE_HINT)


def _index_corpus(root: Path) -> dict[str, list[tuple[Path, float]]]:
    """``{speaker: [(path, duration), ...]}`` in corpus order.

    Order is chapter-then-utterance (``sorted`` over the tree), so taking a
    consecutive slice yields a continuous stretch of one reading session
    wherever possible -- the closest this corpus gets to a person talking for a
    minute. Durations come from file headers; no audio is decoded.
    """
    index: dict[str, list[tuple[Path, float]]] = defaultdict(list)
    for flac in sorted(root.rglob("*.flac")):
        speaker_dir = flac.parent.parent
        if speaker_dir.parent != root:
            continue  # not the expected <speaker>/<chapter>/<utt> depth
        index[speaker_dir.name].append((flac, _utterance_duration(flac)))
    return dict(index)


def _take_span(
    utterances: list[tuple[Path, float]], target_sec: float, rng: random.Random
) -> list[Path] | None:
    """A consecutive run of utterances totalling at least ``target_sec``.

    Consecutive rather than random so the concatenation sounds like one person
    speaking continuously rather than a shuffled montage -- the montage would
    add artificial acoustic jumps that a diarizer could latch onto, which would
    flatter the result for the wrong reason.

    The start is randomised so different scenes drawing the same speaker do not
    all reuse the same opening utterances.
    """
    n = len(utterances)
    if n == 0:
        return None
    for start in rng.sample(range(n), n):
        picked: list[Path] = []
        total = 0.0
        for path, duration in utterances[start:]:
            picked.append(path)
            total += duration
            if total >= target_sec:
                return picked
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-root", type=Path, required=True,
                        help="a split directory, e.g. .../LibriSpeech/test-clean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-src", type=int, default=3)
    parser.add_argument("--num-scenes", type=int, default=50)
    parser.add_argument("--per-speaker-sec", type=float, default=60.0,
                        help="audio each speaker contributes; scene length follows "
                             "from placement (printed below)")
    parser.add_argument("--overlap", type=float, default=0.3,
                        help="used to PREDICT scene length and to check the middle "
                             "speaker is not starved; the value that actually applies "
                             "comes from the eval config -- keep the two in sync")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    predicted = (1 + (args.n_src - 1) * (1 - args.overlap)) * args.per_speaker_sec
    print(
        f"per speaker: {args.per_speaker_sec:.0f}s  ->  predicted scene length "
        f"(chain, overlap={args.overlap}): {predicted:.0f}s = {predicted/60:.1f} min"
    )
    if predicted < 90:
        print(
            "  WARNING: under ~90 s. The whole point of this corpus is to reach the "
            "duration diarizers are built for; short scenes reproduce the collapse."
        )
    if args.n_src >= 3 and args.overlap >= 0.5:
        raise SystemExit(
            f"--overlap {args.overlap} starves the middle speaker under chain "
            "placement: at 0.5 its solo window closes exactly (measured 32/1/33 s "
            "across three speakers), which is the same starvation that made "
            "pyannote return 2 clusters. Use < 0.5 -- 0.3 gives 45/26/45 s."
        )

    rng = random.Random(args.seed)
    index = _index_corpus(args.librispeech_root)
    if not index:
        raise SystemExit(f"no .flac files found under {args.librispeech_root}")

    usable = {
        speaker: utterances for speaker, utterances in index.items()
        if sum(d for _, d in utterances) >= args.per_speaker_sec
    }
    print(
        f"corpus: {len(index)} speakers indexed, {len(usable)} with at least "
        f"{args.per_speaker_sec:.0f}s of audio"
    )
    if len(usable) < args.n_src:
        raise SystemExit(
            f"only {len(usable)} usable speakers but n_src={args.n_src}. Lower "
            "--per-speaker-sec, or use a split with more audio per speaker."
        )

    fieldnames = ["mixture_ID"]
    for k in range(1, args.n_src + 1):
        fieldnames += [f"source_{k}_path", f"source_{k}_gain"]

    speaker_ids = sorted(usable)
    rows: list[dict] = []
    utt_counts: list[int] = []
    attempts = 0
    while len(rows) < args.num_scenes:
        attempts += 1
        if attempts > args.num_scenes * 50:
            raise SystemExit(
                f"gave up after {attempts} attempts with {len(rows)} scenes -- "
                "constraints are too tight for this split."
            )
        # Distinct speakers within a scene; speakers may recur ACROSS scenes,
        # which is what LibriMix itself does.
        chosen = rng.sample(speaker_ids, args.n_src)
        row: dict[str, str] = {}
        ok = True
        for k, speaker in enumerate(chosen, start=1):
            span = _take_span(usable[speaker], args.per_speaker_sec, rng)
            if span is None:
                ok = False
                break
            # Paths relative to the LibriSpeech root's parent -- the shape
            # _resolve_source_path already handles.
            rel = [str(p.relative_to(args.librispeech_root.parent)) for p in span]
            row[f"source_{k}_path"] = "|".join(rel)
            row[f"source_{k}_gain"] = "1.0"
            utt_counts.append(len(rel))
        if not ok:
            continue
        row["mixture_ID"] = f"long_{len(rows):05d}_" + "-".join(chosen)
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    mean_utts = sum(utt_counts) / len(utt_counts)
    print(
        f"wrote {len(rows)} scenes to {args.output}\n"
        f"  {mean_utts:.1f} utterances concatenated per speaker on average "
        f"(min {min(utt_counts)}, max {max(utt_counts)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
