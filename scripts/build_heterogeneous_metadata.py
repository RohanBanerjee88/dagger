#!/usr/bin/env python3
"""Build scenes where a speaker's SOLO region and their OVERLAPPED speech come
from different recordings (CLAUDE.md Phase 2, "experiment 2").

Why this corpus exists
----------------------
In stock LibriMix each speaker contributes one utterance, and
``schedule_solo_then_overlap`` splits that single recording in two: the head
becomes the solo slot (which enrollment reads), the tail becomes the overlap
(which ``G`` extracts and coarse-to-fine re-embeds). Same session, same mic,
same prosody -- so the overlap portion contains nothing about the speaker that
the solo clip doesn't already, apart from extra duration. That is a structural
ceiling on how much embedding refinement could ever recover, and it means the
2026-08-03 finding ("refinement costs 0.2-1.1 dB") was measured in a regime
where refinement had no information available to add.

This generator gives each speaker TWO utterances, written ``pathA|pathB`` and
concatenated by :class:`dagger.data.librimix.LibriMixDataset`. Utterance A fills
the solo slot; B is the overlapped speech. Now the enrollment genuinely does not
describe what is being extracted, which is the condition under which refinement
should pay -- a prediction stated before the experiment, not after it.

The matched control
-------------------
Heterogeneous scenes differ from the stock ones in THREE ways at once: pairing,
scene length, and ``min_solo``. Comparing them against the existing 5-speaker
results would confound all three. So this script emits both arms at identical
geometry, switched by ``--pairing``:

* ``different-chapter`` (treatment) -- A and B from different LibriSpeech
  chapters of the same speaker: different reading sessions.
* ``same-chapter`` (control) -- A and B adjacent within one chapter: as close to
  one continuous recording as two utterances get.

Run both, change nothing else, and the only variable is whether the enrolled
recording is the one being extracted.

Keeping the solo slot inside utterance A
----------------------------------------
``schedule_solo_then_overlap`` takes ``min(min_solo, len)`` from the head of the
concatenation, so if ``min_solo`` exceeded ``len(A)`` the solo region would spill
into B and the arms would blur. ``--solo-sec`` therefore filters A to utterances
at least that long, and the matching config sets ``dataset.min_solo_ms`` to the
same value -- so the solo slot is always a prefix of A, for every speaker.
``--solo-max-sec`` keeps A from being much longer than needed, since the
leftover ``len(A) - min_solo`` lands in the overlap zone and dilutes the
contrast.

    python scripts/build_heterogeneous_metadata.py \\
        --librispeech-root /kaggle/working/data/LibriSpeech/test-clean \\
        --output /kaggle/working/data/metadata/Libri3Mix/libri3mix_hetero_5spk.csv \\
        --n-src 5 --num-scenes 150 --pairing different-chapter
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


def _index_corpus(root: Path) -> dict[str, dict[str, list[tuple[Path, float]]]]:
    """``{speaker: {chapter: [(path, duration), ...]}}``, chapters sorted by name.

    LibriSpeech layout is ``<split>/<speaker>/<chapter>/<utt>.flac``. Durations
    are read from headers only -- no audio is decoded, so indexing test-clean
    takes seconds rather than minutes.
    """
    index: dict[str, dict[str, list[tuple[Path, float]]]] = defaultdict(lambda: defaultdict(list))
    for flac in sorted(root.rglob("*.flac")):
        chapter_dir = flac.parent
        speaker_dir = chapter_dir.parent
        if speaker_dir.parent != root:
            continue  # not the expected <speaker>/<chapter>/<utt> depth
        index[speaker_dir.name][chapter_dir.name].append((flac, _utterance_duration(flac)))
    return {speaker: dict(chapters) for speaker, chapters in index.items()}


def _pick_pair(
    chapters: dict[str, list[tuple[Path, float]]],
    rng: random.Random,
    *,
    pairing: str,
    solo_sec: float,
    solo_max_sec: float,
    overlap_min_sec: float,
    overlap_max_sec: float,
) -> tuple[Path, Path] | None:
    """One speaker's (A, B): A long enough to hold the whole solo slot, B the
    overlapped speech. ``None`` when this speaker can't satisfy the constraints.
    """
    candidates_a = [
        (chapter, path)
        for chapter, utterances in chapters.items()
        for path, duration in utterances
        if solo_sec <= duration <= solo_max_sec
    ]
    if not candidates_a:
        return None
    rng.shuffle(candidates_a)

    for chapter_a, path_a in candidates_a:
        if pairing == "different-chapter":
            pool = [
                path
                for chapter, utterances in chapters.items()
                if chapter != chapter_a
                for path, duration in utterances
                if overlap_min_sec <= duration <= overlap_max_sec
            ]
        else:  # same-chapter: the control arm
            pool = [
                path
                for path, duration in chapters[chapter_a]
                if path != path_a and overlap_min_sec <= duration <= overlap_max_sec
            ]
        if pool:
            return path_a, rng.choice(pool)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-root", type=Path, required=True,
                        help="a split directory, e.g. .../LibriSpeech/test-clean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-src", type=int, default=5)
    parser.add_argument("--num-scenes", type=int, default=150)
    parser.add_argument("--pairing", choices=("different-chapter", "same-chapter"),
                        default="different-chapter",
                        help="different-chapter = treatment; same-chapter = matched control")
    parser.add_argument("--solo-sec", type=float, default=7.0,
                        help="minimum length of utterance A; set dataset.min_solo_ms to this")
    parser.add_argument("--solo-max-sec", type=float, default=9.0,
                        help="cap on A, so its leftover doesn't dilute the overlap zone")
    parser.add_argument("--overlap-min-sec", type=float, default=6.0,
                        help="minimum length of utterance B")
    parser.add_argument("--overlap-max-sec", type=float, default=9.0,
                        help="maximum length of utterance B; the BAND matters more than "
                             "either edge -- see the taper check below")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.solo_max_sec < args.solo_sec:
        raise SystemExit("--solo-max-sec must be >= --solo-sec")
    if args.overlap_max_sec < args.overlap_min_sec:
        raise SystemExit("--overlap-max-sec must be >= --overlap-min-sec")

    # The designated solo slot is not the only solo region in the scene. In the
    # overlap zone, tails end at different times, so whichever speaker's tail
    # outlasts the rest is ALONE for the difference -- a second solo run, made
    # of utterance B. `select_topk_solo_clips` takes the LONGEST run, so if that
    # taper exceeds `solo_sec`, enrollment reads B instead of A and the
    # treatment arm silently becomes the control.
    #
    #   tail_i          = (len(A_i) - solo_sec) + len(B_i)
    #   worst-case taper = spread(len(A)) + spread(len(B))
    #
    # Keeping both duration bands narrow is what bounds it. This is a hard error
    # rather than a warning: a run that violates it produces results that look
    # fine and mean nothing.
    worst_taper = (args.solo_max_sec - args.solo_sec) + (args.overlap_max_sec - args.overlap_min_sec)
    print(
        f"worst-case taper solo: {worst_taper:.1f}s vs designated slot {args.solo_sec:.1f}s"
    )
    if worst_taper >= args.solo_sec:
        raise SystemExit(
            f"taper solo could reach {worst_taper:.1f}s, at or above the {args.solo_sec:.1f}s "
            "designated slot -- enrollment could then read utterance B and the arms would be "
            "indistinguishable. Narrow --solo-max-sec/--overlap-max-sec (or raise --solo-sec) "
            f"so that (solo_max - solo_sec) + (overlap_max - overlap_min) < {args.solo_sec:.1f}."
        )

    rng = random.Random(args.seed)
    index = _index_corpus(args.librispeech_root)
    if not index:
        raise SystemExit(f"no .flac files found under {args.librispeech_root}")

    usable = {
        speaker: chapters for speaker, chapters in index.items()
        if _pick_pair(
            chapters, random.Random(args.seed), pairing=args.pairing,
            solo_sec=args.solo_sec, solo_max_sec=args.solo_max_sec,
            overlap_min_sec=args.overlap_min_sec, overlap_max_sec=args.overlap_max_sec,
        ) is not None
    }
    print(
        f"corpus: {len(index)} speakers indexed, {len(usable)} usable under "
        f"pairing={args.pairing}, solo {args.solo_sec}-{args.solo_max_sec}s, "
        f"overlap {args.overlap_min_sec}-{args.overlap_max_sec}s"
    )
    if len(usable) < args.n_src:
        raise SystemExit(
            f"only {len(usable)} usable speakers but n_src={args.n_src}. Loosen "
            "--solo-sec / --solo-max-sec / --overlap-min-sec, or use a larger split "
            "(train-clean-360 has far more speakers than test-clean)."
        )

    fieldnames = ["mixture_ID"]
    for k in range(1, args.n_src + 1):
        fieldnames += [f"source_{k}_path", f"source_{k}_gain"]

    rows = []
    speaker_ids = sorted(usable)
    skipped = 0
    while len(rows) < args.num_scenes:
        chosen = rng.sample(speaker_ids, args.n_src)
        row = {}
        parts = []
        for k, speaker in enumerate(chosen, start=1):
            pair = _pick_pair(
                usable[speaker], rng, pairing=args.pairing,
                solo_sec=args.solo_sec, solo_max_sec=args.solo_max_sec,
                overlap_min_sec=args.overlap_min_sec, overlap_max_sec=args.overlap_max_sec,
            )
            if pair is None:
                break
            path_a, path_b = pair
            # Paths relative to the LibriSpeech root, the shape
            # _resolve_source_path already handles.
            rel_a = path_a.relative_to(args.librispeech_root.parent)
            rel_b = path_b.relative_to(args.librispeech_root.parent)
            row[f"source_{k}_path"] = f"{rel_a}|{rel_b}"
            row[f"source_{k}_gain"] = "1.0"
            parts.append(speaker)
        else:
            row["mixture_ID"] = f"hetero_{args.pairing}_{len(rows):05d}_" + "-".join(parts)
            rows.append(row)
            continue
        skipped += 1
        if skipped > 20 * args.num_scenes:
            raise SystemExit("could not assemble enough scenes; loosen the constraints")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"wrote {len(rows)} scenes to {args.output}\n"
        f"Set dataset.min_solo_ms = {args.solo_sec * 1000:.0f} in the matching config so the "
        "solo slot stays inside utterance A."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
