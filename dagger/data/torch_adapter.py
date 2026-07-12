"""Fixed-length, batchable training crops on top of any :class:`SceneDataset`.

Phase 0's :class:`~dagger.data.base.SceneDataset` is scene-at-a-time and
variable-length by design (fine for eval, not for ``DataLoader`` batching).
:func:`build_scene_crop_dataset` materializes the (small, ``limit``-capped)
dataset once, precomputes oracle activity/solo/overlap and -- if an encoder is
given -- each speaker's enrollment embedding from the *full* scene (cropping
to a training segment first could cut a solo region too short to enroll
from), then serves random fixed-length crops per ``__getitem__``.

Enrollment can be a long-running pass over the whole dataset. A scene where
any speaker has no solo region (or only fragments shorter than
``min_clip_ms``) raises inside :func:`~dagger.enroll.topk.enroll_speaker`;
rather than aborting the whole dataset pass, that scene is skipped, logged
immediately, and counted in a summary printed once preparation finishes.
"""

from __future__ import annotations

import numpy as np

from dagger.data.base import Scene, SceneDataset
from dagger.diarize.oracle import activity_matrix, solo_overlap_regions
from dagger.enroll.encoder import SpeakerEncoder
from dagger.enroll.topk import NoSoloRegionError, enroll_speaker
from dagger.reconstruct.stitch import crossfade_windows


def _overlap_runs(overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, cumulative_lengths)`` of contiguous overlap runs in a 0/1 mask."""
    padded = np.concatenate([[0], np.asarray(overlap).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    starts, ends = edges[0::2], edges[1::2]
    return starts, np.cumsum(ends - starts)


def build_scene_crop_dataset(
    dataset: SceneDataset,
    segment_seconds: float,
    encoder: SpeakerEncoder | None = None,
    enroll_k: int = 3,
    fade: int = 0,
    seed: int | None = None,
    require_overlap: bool = False,
):
    """Build a ``torch.utils.data.Dataset`` of fixed-length training crops.

    Imports ``torch`` inside this function, so ``import dagger.data.torch_adapter``
    never requires the ``[ml]`` extra -- only calling this function does. Each
    item is a dict of ``[T]``/``[S,T]`` tensors: ``mixture``, ``sources``,
    ``activity``, ``solo``, ``overlap``, ``w_overlap`` (the per-speaker
    ``w_Oi`` crossfade window from :func:`~dagger.reconstruct.stitch.crossfade_windows`,
    i.e. exactly the weighting the proposed extractor is scored against at
    inference time), and (if ``encoder`` was given) ``embeddings`` (``[S, D]``,
    the frozen per-speaker enrollment).

    Crops are *overlap-centered*: each ``__getitem__`` picks a random overlap
    sample (uniform over overlap samples, via precomputed run boundaries) and
    centers the crop on it, falling back to a uniform-random start when the
    scene has no overlap. Uniform starts mostly land on solo stretches, where
    the extractor's windowed loss has no target — see the Phase 1 training
    notes in ``scripts/train_phase1.py``.

    ``require_overlap=True`` drops scenes with no overlap at all (logged, like
    the enrollment skips): they contribute zero extractor training signal.
    """
    import torch

    def _prepare(scene: Scene):
        n = scene.mixture.shape[0]
        activity, speakers = activity_matrix(
            scene.segments, num_samples=n, sample_rate=scene.sample_rate,
            speakers=scene.speakers,
        )
        solo, overlap = solo_overlap_regions(activity)
        w_overlap = np.stack(
            [crossfade_windows(solo[i], activity[i], fade=fade)[1] for i in range(len(speakers))],
            axis=0,
        )
        embeddings = None
        if encoder is not None:
            embeddings = np.stack(
                [
                    enroll_speaker(
                        scene.mixture, solo[i], activity[i], scene.sample_rate,
                        encoder, k=enroll_k,
                    ).embedding
                    for i in range(len(speakers))
                ],
                axis=0,
            )
        # Store compactly: __getitem__ serves float32 tensors regardless, and
        # the masks are 0/1 -- keeping the float64 originals (plus the whole
        # Scene object) is ~112 bytes/sample and OOMs the host RAM near
        # ~2000 scenes. float32 audio + uint8 masks is ~35 bytes/sample.
        return {
            "mixture": scene.mixture.astype(np.float32),
            "sources": scene.sources.astype(np.float32),
            "sample_rate": scene.sample_rate,
            "activity": activity.astype(np.uint8),
            "solo": solo.astype(np.uint8),
            "overlap": overlap.astype(np.uint8),
            "w_overlap": w_overlap.astype(np.float32),
            "embeddings": embeddings,
            "overlap_runs": _overlap_runs(overlap),
        }

    prepared = []
    skipped_no_overlap: list[str] = []
    skipped: list[tuple[str, str]] = []  # (scene name, reason) -- enrollment can be
    # a slow, long-running pass over the whole dataset (one encoder forward pass per
    # solo clip per speaker per scene); a single speaker with no solo region -- or only
    # solo runs shorter than min_clip_ms -- must not abort that entire pass, so each
    # scene's enrollment is attempted independently and skip failures are logged both
    # as they happen and in the summary below, rather than only surfacing at the end.
    for scene in dataset:
        try:
            item = _prepare(scene)
        except NoSoloRegionError as exc:
            # Only the benign "no usable solo audio" case is skip-worthy; a plain
            # ValueError (e.g. the overlap-contamination guard in dagger.enroll.topk)
            # indicates a real bug and must propagate, not be silently swallowed here.
            reason = str(exc)
            skipped.append((scene.name, reason))
            print(f"[enroll] skipping scene {scene.name!r}: {reason}")
            continue
        if require_overlap and item["overlap"].sum() == 0:
            skipped_no_overlap.append(scene.name)
            print(f"[crops] skipping scene {scene.name!r}: no overlap anywhere.")
            continue
        prepared.append(item)

    if skipped_no_overlap:
        print(
            f"[crops] skipped {len(skipped_no_overlap)} scene(s) with no overlap "
            f"(nothing for the extractor to learn from): {skipped_no_overlap}"
        )
    if skipped:
        print(
            f"[enroll] skipped {len(skipped)}/{len(skipped) + len(prepared)} scenes "
            f"during enrollment (see per-scene messages above): "
            f"{[name for name, _ in skipped]}"
        )

    rng = np.random.default_rng(seed)

    class _SceneCropDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return len(prepared)

        def __getitem__(self, idx: int):
            item = prepared[idx]
            n = item["mixture"].shape[0]
            seg = int(round(segment_seconds * item["sample_rate"]))
            run_starts, run_cumlen = item["overlap_runs"]
            if n <= seg:
                start = 0
            elif run_cumlen.size:
                # Center the crop on a random overlap sample (uniform over all
                # overlap samples: pick a global overlap index, map it into its
                # run via the cumulative lengths).
                k = int(rng.integers(0, run_cumlen[-1]))
                run = int(np.searchsorted(run_cumlen, k, side="right"))
                offset_in_run = k - (int(run_cumlen[run - 1]) if run else 0)
                t = int(run_starts[run]) + offset_in_run
                start = int(np.clip(t - seg // 2, 0, n - seg))
            else:
                start = int(rng.integers(0, n - seg + 1))
            end = start + seg

            def crop_1d(x: np.ndarray) -> np.ndarray:
                c = np.asarray(x)[start:end]
                if c.shape[0] < seg:
                    c = np.pad(c, (0, seg - c.shape[0]))
                return c

            def crop_2d(x: np.ndarray) -> np.ndarray:
                return np.stack([crop_1d(row) for row in x], axis=0)

            sample = {
                "mixture": torch.as_tensor(crop_1d(item["mixture"]), dtype=torch.float32),
                "sources": torch.as_tensor(crop_2d(item["sources"]), dtype=torch.float32),
                "activity": torch.as_tensor(crop_2d(item["activity"]), dtype=torch.float32),
                "solo": torch.as_tensor(crop_2d(item["solo"]), dtype=torch.float32),
                "overlap": torch.as_tensor(crop_1d(item["overlap"]), dtype=torch.float32),
                "w_overlap": torch.as_tensor(crop_2d(item["w_overlap"]), dtype=torch.float32),
            }
            if item["embeddings"] is not None:
                sample["embeddings"] = torch.as_tensor(
                    item["embeddings"], dtype=torch.float32
                )
            return sample

    return _SceneCropDataset()
