#!/usr/bin/env python3
"""Phase 1 training entrypoint: trains the proposed extractor G or the blind baseline.

Two independently-checkpointed systems, selected with ``--system``:

* ``proposed`` -- trains :func:`dagger.extract.tfgridnet_crossattn.
  build_tfgridnet_crossattn_module`'s ``nn.Module`` directly (not through the
  inference-only :class:`~dagger.extract.tfgridnet_crossattn.TFGridNetCrossAttnExtractor`
  wrapper). Per speaker, scores ``G(x_O, e_bar_i)`` against the clean source,
  weighted by the same ``w_Oi`` crossfade window used at inference time
  (:func:`dagger.reconstruct.stitch.crossfade_windows`) -- so training matches
  inference exactly, including the known Phase 1 hard-mask-input limitation
  (CLAUDE.md §5, deferred to Phase 2; see the comment in
  ``dagger/reconstruct/stitch.py``).
* ``blind`` -- trains :func:`dagger.extract.blind.build_blind_separator_module`
  on the full mixture against all clean sources jointly via permutation
  -invariant loss (:mod:`dagger.losses.pit`). No embeddings needed.

Both trainings use frozen oracle diarization and (for ``proposed``) a frozen
speaker encoder (CLAUDE.md §3: "Freeze pretrained weights first").

``train.init_checkpoint`` (``proposed`` only, optional): warm-start from an
existing checkpoint's weights instead of random init -- e.g. Phase 2 fine-tuning
the Phase 1 checkpoint on ``dataset.placement: scheduled`` scenes so ``G`` gets
real depth-3 overlap exposure. Must match the current run's ``extractor`` config
(same architecture) or ``load_state_dict`` raises.

``dataset`` (``proposed`` only) may be a single dataset config (as always) or a
*list* of them -- e.g. one entry per overlap depth for curriculum training
across a mix of depths in one run (CLAUDE.md Phase 2 "Stage 2": a single
fixed-depth fine-tune keeps reproducing the same raise-the-floor/narrow-the-gap
pattern regardless of which depth it targets). Each entry builds its own
loader via the existing, unmodified :func:`~dagger.data.build_dataset` /
:func:`~dagger.data.torch_adapter.build_scene_crop_dataset`; one shared model
and optimizer see batches drawn from all loaders, interleaved in a shuffled
order each epoch. A single-entry list (including a bare dict, treated as one)
reduces to exactly the single-dataset behavior below -- this is one code path,
not a special case.

Reproduce with::

    DAGGER_DATA_ROOT=/mnt/data python scripts/train_phase1.py \\
        --config configs/phase1_smoke.yaml --system proposed
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dagger.data import build_dataset
from dagger.data.paths import load_env
from dagger.data.torch_adapter import build_scene_crop_dataset


def _device(preferred: str | None) -> str:
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def _save_checkpoint(
    model, model_config: dict, system: str, path: Path, trained_n_src: list | None = None,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"state_dict": model.state_dict(), "model_config": model_config,
             "phase": "1", "system": system}
    if trained_n_src is not None:
        state["trained_n_src"] = trained_n_src
    torch.save(state, path)


def _grad_norm_summary(norms: list[float], clip: float | None) -> str:
    """Epoch-log suffix describing pre-clip gradient norms.

    Exists to validate the ``train.grad_clip`` threshold empirically: healthy
    is a median comfortably below the clip with a small clipped-% (outlier
    protection only); a median at/above the clip means every step is being
    rescaled -- i.e. the clip is acting as a hidden LR reduction -- so raise it.
    """
    if not norms:
        return ""
    med = statistics.median(norms)
    mx = max(norms)
    if clip is None:
        return f"  grad_norm: median={med:.2f} max={mx:.2f} (clip off)"
    frac = 100.0 * sum(n > clip for n in norms) / len(norms)
    return f"  grad_norm: median={med:.2f} max={mx:.2f} clipped={frac:.0f}%"


def _checkpoint_path(cfg: dict, system: str) -> Path:
    """Checkpoint path for ``system``, derived deterministically from config.

    Strips any existing ``proposed_``/``blind_`` prefix off the configured
    ``checkpoint_out`` stem and prepends ``system`` -- so ``proposed`` and
    ``blind`` runs of the *same* config always land at different paths,
    regardless of what the config's filename happens to contain (previously
    this was a fragile ``"proposed"`` -> ``"blind"`` string substitution that
    silently collided for filenames like ``smoke.pt`` with no "proposed"
    substring to replace).
    """
    base = Path(cfg["train"]["checkpoint_out"])
    stem = base.stem
    for prefix in ("proposed_", "blind_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    return base.parent / f"{system}_{stem}{base.suffix}"


def train_proposed(cfg: dict, device: str) -> None:
    import random

    import torch

    from dagger.enroll.encoder import TitaNetEncoder
    from dagger.extract.tfgridnet_crossattn import build_tfgridnet_crossattn_module
    from dagger.losses.sisdr import si_sdr_loss

    fade = int(round(cfg.get("fade_ms", 0) / 1000.0 * int(cfg["sample_rate"])))
    datasets_cfg = cfg["dataset"] if isinstance(cfg["dataset"], list) else [cfg["dataset"]]
    encoder = TitaNetEncoder(device=device)

    # One loader per dataset entry (e.g. one per overlap depth for curriculum
    # training). A single-entry list -- the common case -- makes this loop
    # build exactly one loader, identical to the pre-curriculum code path.
    loaders: list[tuple[object, "torch.utils.data.DataLoader"]] = []
    for dataset_cfg in datasets_cfg:
        dataset = build_dataset({**cfg, "dataset": dataset_cfg})
        crops = build_scene_crop_dataset(
            dataset,
            segment_seconds=cfg["train"]["segment_seconds"],
            encoder=encoder,
            enroll_k=cfg.get("enroll", {}).get("k", 3),
            fade=fade,
            require_overlap=True,  # a no-overlap scene has nothing for G to learn from
        )
        loader = torch.utils.data.DataLoader(
            crops, batch_size=cfg["train"]["batch_size"], shuffle=True
        )
        loaders.append((dataset_cfg.get("n_src"), loader))
        print(f"[proposed] built loader for n_src={dataset_cfg.get('n_src')}: {len(crops)} scenes")

    model = build_tfgridnet_crossattn_module(cfg.get("extractor", {})).to(device)
    init_checkpoint_path = cfg["train"].get("init_checkpoint")
    if init_checkpoint_path:
        # Fine-tune from an existing checkpoint (e.g. Phase 2's "adapt to real
        # depth-3 overlap" run warm-starting from the Phase 1 checkpoint)
        # instead of the default random init. Requires the SAME extractor
        # architecture (cfg["extractor"]) the checkpoint was trained with --
        # load_state_dict raises loudly on a shape mismatch otherwise.
        state = torch.load(init_checkpoint_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        print(f"[proposed] warm-started from {init_checkpoint_path}")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    grad_clip = cfg["train"].get("grad_clip", 5.0)
    checkpoint_every = cfg["train"].get("checkpoint_every", 5)
    checkpoint_out = _checkpoint_path(cfg, "proposed")
    trained_n_src = [n_src for n_src, _ in loaders]

    for epoch in range(cfg["train"]["epochs"]):
        total_loss = 0.0
        n_batches = 0
        grad_norms: list[float] = []

        # Fresh iterators each epoch (re-triggers each DataLoader's own
        # shuffle=True reshuffle). Batches are drawn in a per-epoch-shuffled
        # order interleaved ACROSS loaders, so one epoch mixes depths -- but
        # every individual batch still comes from exactly one loader (so it's
        # internally uniform in num_speakers), which is why no custom
        # collate/padding is needed anywhere. With one loader this reduces to
        # draining it in its own shuffled order, identical to before.
        iters = [iter(loader) for _, loader in loaders]
        draw_order = [idx for idx, (_, loader) in enumerate(loaders) for _ in range(len(loader))]
        random.shuffle(draw_order)

        for idx in draw_order:
            batch = next(iters[idx])
            mixture = batch["mixture"].to(device)
            overlap = batch["overlap"].to(device)
            sources = batch["sources"].to(device)
            w_overlap = batch["w_overlap"].to(device)
            embeddings = batch["embeddings"].to(device)
            num_speakers = sources.shape[1]

            x_o = mixture * overlap  # shared hard-masked x_O, same as inference

            # A (crop, speaker) term is scoreable only if the speaker's overlap
            # window actually intersects the crop: SI-SDR is scale-invariant,
            # so a ~zero windowed target cannot express "output silence" -- it
            # degenerates to -10*log10(eps) (~+80) with a garbage gradient
            # that swamps the real terms. Skip those terms and average the
            # loss over the scoreable ones only.
            windowed_targets = sources * w_overlap  # [B, S, T]
            valid = windowed_targets.pow(2).sum(dim=-1) > 1e-8  # [B, S]
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            optimizer.zero_grad()
            # Backward per speaker so only one extractor graph is alive at a
            # time -- summing all speakers' losses before backward() holds
            # num_speakers full TF-GridNet graphs and OOMs on 16 GB GPUs.
            # Gradients accumulate across backward() calls, so this matches
            # the summed loss exactly.
            batch_loss = 0.0
            for i in range(num_speakers):
                sel = valid[:, i]
                if not bool(sel.any()):
                    continue
                estimate = model(x_o[sel], embeddings[sel, i, :])
                weight = w_overlap[sel, i, :]
                per_item = si_sdr_loss(
                    estimate * weight, sources[sel, i, :] * weight, reduction="none"
                )
                loss = per_item.sum() / n_valid
                loss.backward()
                batch_loss += float(loss.item())
            # A single unclipped step can undo hundreds of good ones (the
            # +1-to-+2 loss spikes in otherwise-descending runs); clip the
            # accumulated gradient once, right before the step. max_norm=inf
            # when disabled -- a no-op that still returns the pre-clip norm
            # for the epoch log.
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip if grad_clip is not None else float("inf"),
            )
            grad_norms.append(float(total_norm))
            optimizer.step()

            total_loss += batch_loss
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        print(
            f"[proposed] epoch {epoch + 1}/{cfg['train']['epochs']}  loss={mean_loss:.4f}"
            f"{_grad_norm_summary(grad_norms, grad_clip)}"
        )
        # Periodic (overwriting) save so an interrupted long run keeps its
        # latest state instead of losing everything to the end-only save.
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(model, cfg.get("extractor", {}), "proposed", checkpoint_out, trained_n_src)
            print(f"[proposed] checkpoint saved @ epoch {epoch + 1} -> {checkpoint_out}")

    _save_checkpoint(model, cfg.get("extractor", {}), "proposed", checkpoint_out, trained_n_src)
    print(f"saved checkpoint to {checkpoint_out}")


def train_blind(cfg: dict, device: str) -> None:
    import torch

    from dagger.extract.blind import build_blind_separator_module
    from dagger.losses.pit import pit_loss

    dataset = build_dataset(cfg)
    crops = build_scene_crop_dataset(
        dataset, segment_seconds=cfg["train"]["segment_seconds"], encoder=None,
    )
    loader = torch.utils.data.DataLoader(
        crops, batch_size=cfg["train"]["batch_size"], shuffle=True
    )

    extractor_cfg = dict(cfg.get("extractor", {}))
    extractor_cfg.setdefault("num_speakers", cfg["dataset"].get("n_src", 2))
    model = build_blind_separator_module(extractor_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    grad_clip = cfg["train"].get("grad_clip", 5.0)
    checkpoint_every = cfg["train"].get("checkpoint_every", 5)
    checkpoint_out = _checkpoint_path(cfg, "blind")

    for epoch in range(cfg["train"]["epochs"]):
        total_loss = 0.0
        n_batches = 0
        grad_norms: list[float] = []
        for batch in loader:
            mixture = batch["mixture"].to(device)
            sources = batch["sources"].to(device)

            optimizer.zero_grad()
            estimates = model(mixture)
            loss = pit_loss(estimates, sources)
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=grad_clip if grad_clip is not None else float("inf"),
            )
            grad_norms.append(float(total_norm))
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        print(
            f"[blind] epoch {epoch + 1}/{cfg['train']['epochs']}  loss={mean_loss:.4f}"
            f"{_grad_norm_summary(grad_norms, grad_clip)}"
        )
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(model, extractor_cfg, "blind", checkpoint_out)
            print(f"[blind] checkpoint saved @ epoch {epoch + 1} -> {checkpoint_out}")

    _save_checkpoint(model, extractor_cfg, "blind", checkpoint_out)
    print(f"saved checkpoint to {checkpoint_out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase1_smoke.yaml")
    parser.add_argument("--system", choices=["proposed", "blind"], default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    load_env()
    cfg = yaml.safe_load(Path(args.config).read_text())
    system = args.system or cfg.get("train", {}).get("system", "proposed")
    device = _device(args.device)
    print(f"training system={system!r} device={device!r} config={args.config}")

    if system == "proposed":
        train_proposed(cfg, device)
    else:
        train_blind(cfg, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
