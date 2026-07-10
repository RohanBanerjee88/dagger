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

Reproduce with::

    DAGGER_DATA_ROOT=/mnt/data python scripts/train_phase1.py \\
        --config configs/phase1_smoke.yaml --system proposed
"""

from __future__ import annotations

import argparse
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
    import torch

    from dagger.enroll.encoder import TitaNetEncoder
    from dagger.extract.tfgridnet_crossattn import build_tfgridnet_crossattn_module
    from dagger.losses.sisdr import si_sdr_loss

    fade = int(round(cfg.get("fade_ms", 0) / 1000.0 * int(cfg["sample_rate"])))
    dataset = build_dataset(cfg)
    encoder = TitaNetEncoder(device=device)
    crops = build_scene_crop_dataset(
        dataset,
        segment_seconds=cfg["train"]["segment_seconds"],
        encoder=encoder,
        enroll_k=cfg.get("enroll", {}).get("k", 3),
        fade=fade,
    )
    loader = torch.utils.data.DataLoader(
        crops, batch_size=cfg["train"]["batch_size"], shuffle=True
    )

    model = build_tfgridnet_crossattn_module(cfg.get("extractor", {})).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    for epoch in range(cfg["train"]["epochs"]):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            mixture = batch["mixture"].to(device)
            overlap = batch["overlap"].to(device)
            sources = batch["sources"].to(device)
            w_overlap = batch["w_overlap"].to(device)
            embeddings = batch["embeddings"].to(device)
            num_speakers = sources.shape[1]

            x_o = mixture * overlap  # shared hard-masked x_O, same as inference

            optimizer.zero_grad()
            # Backward per speaker so only one extractor graph is alive at a
            # time -- summing all speakers' losses before backward() holds
            # num_speakers full TF-GridNet graphs and OOMs on 16 GB GPUs.
            # Gradients accumulate across backward() calls, so this matches
            # the summed loss exactly.
            batch_loss = 0.0
            for i in range(num_speakers):
                estimate = model(x_o, embeddings[:, i, :])
                weight = w_overlap[:, i, :]
                loss = si_sdr_loss(estimate * weight, sources[:, i, :] * weight) / num_speakers
                loss.backward()
                batch_loss += float(loss.item())
            optimizer.step()

            total_loss += batch_loss
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        print(f"[proposed] epoch {epoch + 1}/{cfg['train']['epochs']}  loss={mean_loss:.4f}")

    checkpoint_out = _checkpoint_path(cfg, "proposed")
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_config": cfg.get("extractor", {}),
         "phase": "1", "system": "proposed"},
        checkpoint_out,
    )
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

    for epoch in range(cfg["train"]["epochs"]):
        total_loss = 0.0
        n_batches = 0
        for batch in loader:
            mixture = batch["mixture"].to(device)
            sources = batch["sources"].to(device)

            optimizer.zero_grad()
            estimates = model(mixture)
            loss = pit_loss(estimates, sources)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            n_batches += 1

        mean_loss = total_loss / max(n_batches, 1)
        print(f"[blind] epoch {epoch + 1}/{cfg['train']['epochs']}  loss={mean_loss:.4f}")

    checkpoint_out = _checkpoint_path(cfg, "blind")
    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "model_config": extractor_cfg,
         "phase": "1", "system": "blind"},
        checkpoint_out,
    )
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
