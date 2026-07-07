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


def build_scene_crop_dataset(
    dataset: SceneDataset,
    segment_seconds: float,
    encoder: SpeakerEncoder | None = None,
    enroll_k: int = 3,
    fade: int = 0,
    seed: int | None = None,
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
        return {
            "scene": scene, "activity": activity, "solo": solo,
            "overlap": overlap, "w_overlap": w_overlap, "embeddings": embeddings,
        }

    prepared = []
    skipped: list[tuple[str, str]] = []  # (scene name, reason) -- enrollment can be
    # a slow, long-running pass over the whole dataset (one encoder forward pass per
    # solo clip per speaker per scene); a single speaker with no solo region -- or only
    # solo runs shorter than min_clip_ms -- must not abort that entire pass, so each
    # scene's enrollment is attempted independently and skip failures are logged both
    # as they happen and in the summary below, rather than only surfacing at the end.
    for scene in dataset:
        try:
            prepared.append(_prepare(scene))
        except NoSoloRegionError as exc:
            # Only the benign "no usable solo audio" case is skip-worthy; a plain
            # ValueError (e.g. the overlap-contamination guard in dagger.enroll.topk)
            # indicates a real bug and must propagate, not be silently swallowed here.
            reason = str(exc)
            skipped.append((scene.name, reason))
            print(f"[enroll] skipping scene {scene.name!r}: {reason}")

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
            scene: Scene = item["scene"]
            n = scene.mixture.shape[0]
            seg = int(round(segment_seconds * scene.sample_rate))
            start = 0 if n <= seg else int(rng.integers(0, n - seg + 1))
            end = start + seg

            def crop_1d(x: np.ndarray) -> np.ndarray:
                c = np.asarray(x)[start:end]
                if c.shape[0] < seg:
                    c = np.pad(c, (0, seg - c.shape[0]))
                return c

            def crop_2d(x: np.ndarray) -> np.ndarray:
                return np.stack([crop_1d(row) for row in x], axis=0)

            sample = {
                "mixture": torch.as_tensor(crop_1d(scene.mixture), dtype=torch.float32),
                "sources": torch.as_tensor(crop_2d(scene.sources), dtype=torch.float32),
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
