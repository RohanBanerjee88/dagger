#!/usr/bin/env python3
"""Synthesize a 4/5-speaker LibriMix metadata CSV from an existing 2/3-speaker one.

Phase 2's depth-stratified experiment only has real evidence up to depth 3
(CLAUDE.md's Phase 2 status notes: the accumulation-specific gap between
coarse_to_fine/gated_deflation/ungated_deflation is real but narrow there).
Testing deeper overlap needs 4-5 speaker scenes, but nothing in
``dagger/data/mixing.py`` or the reconstruction/gate/refine modules is capped
at 3 -- they all key off ``activity.shape[0]``. So this is purely a metadata
problem: a LibriMix CSV row is just N independent (speaker, utterance, gain)
triples (``dagger/data/librimix.py`` reads ``source_{k}_path``/
``source_{k}_gain`` generically). Rather than pulling in the official
LibriMix generator for n_src=4/5, this borrows extra (path, gain) pairs from
OTHER rows of the same CSV -- same LibriSpeech audio already on disk, no new
corpus, deterministic given ``--seed``.

Usage::

    python scripts/extend_librimix_metadata.py \\
        --input metadata/Libri3Mix/libri3mix_test.csv \\
        --output metadata/Libri3Mix/libri3mix_test_x5.csv \\
        --target-n-src 5
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from pathlib import Path

SOURCE_PATH_RE = re.compile(r"^source_(\d+)_path$")
MAX_DONOR_ATTEMPTS = 200


def _speaker_id(path: str) -> str:
    """LibriSpeech layout is ``.../<split>/<speaker>/<chapter>/<utt>.flac`` --
    the speaker id is the directory two levels up from the file."""
    return Path(path).parent.parent.name


def _detect_existing_n_src(fieldnames: list[str]) -> int:
    found = {int(m.group(1)) for name in fieldnames if (m := SOURCE_PATH_RE.match(name))}
    if not found:
        raise ValueError(f"no source_<k>_path columns found in {fieldnames!r}")
    if found != set(range(1, len(found) + 1)):
        raise ValueError(f"source_<k>_path columns aren't a contiguous 1..n range: {sorted(found)}")
    return len(found)


def extend_rows(rows: list[dict], existing_n_src: int, target_n_src: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    # Donor pool: every (path, gain) cell across all rows and all existing slots.
    pool: list[tuple[str, str]] = [
        (row[f"source_{k}_path"], row.get(f"source_{k}_gain", "1.0"))
        for row in rows
        for k in range(1, existing_n_src + 1)
    ]

    out_rows = []
    for row in rows:
        row = dict(row)
        used_speakers = {
            _speaker_id(row[f"source_{k}_path"]) for k in range(1, existing_n_src + 1)
        }
        for slot in range(existing_n_src + 1, target_n_src + 1):
            for _attempt in range(MAX_DONOR_ATTEMPTS):
                path, gain = rng.choice(pool)
                spk = _speaker_id(path)
                if spk not in used_speakers:
                    break
            else:
                raise RuntimeError(
                    f"couldn't find a distinct-speaker donor for {row.get('mixture_ID')!r} "
                    f"slot {slot} after {MAX_DONOR_ATTEMPTS} attempts -- pool too small/homogeneous"
                )
            used_speakers.add(spk)
            row[f"source_{slot}_path"] = path
            row[f"source_{slot}_gain"] = gain
        row["mixture_ID"] = f"{row.get('mixture_ID', '')}_x{target_n_src}"
        out_rows.append(row)
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-n-src", required=True, type=int)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()

    with open(args.input, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    existing_n_src = _detect_existing_n_src(fieldnames)
    if args.target_n_src <= existing_n_src:
        raise ValueError(
            f"--target-n-src ({args.target_n_src}) must exceed the input's n_src ({existing_n_src})"
        )

    out_rows = extend_rows(rows, existing_n_src, args.target_n_src, args.seed)

    new_cols = [
        f"source_{k}_{field}"
        for k in range(existing_n_src + 1, args.target_n_src + 1)
        for field in ("path", "gain")
    ]
    out_fieldnames = fieldnames + [c for c in new_cols if c not in fieldnames]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(
        f"wrote {len(out_rows)} rows ({existing_n_src} -> {args.target_n_src} speakers) "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
