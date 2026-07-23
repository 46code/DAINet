"""Synthesize a slim ``model_final.pt`` from a full ``latest.pt``.

When a run is interrupted (killed mid-epoch), the trainer never reaches its
end-of-fit step and so never writes ``model_final.pt`` — only the per-iter
``latest.pt`` (full training state: model + optimizer + scaler + EMA) survives.
A blind rename is wrong: ``latest.pt`` is a different, much larger schema than
the slim ``{"model", "config"}`` that every inference/report path expects
(``load_model_from_checkpoint`` reads ``ckpt["model"]``; a 1.5 GB "final" would
be an inconsistent oddball next to its ~0.5 GB siblings).

This tool reproduces the trainer's end-of-fit convention
(``training/trainer.py`` final-save): ``model_final.pt`` is written from the
**EMA** weights when EMA is enabled (they generalise better), else from the
live weights. It loads ``latest.pt``, overlays the EMA shadow tensors onto the
full model state dict (exactly what ``EMAModel.average_parameters`` +
``state_dict()`` would have produced), and saves a slim light checkpoint.

Usage:
    python scripts/finalize_checkpoint.py \
        --latest runs/dainet_full/checkpoints/latest.pt \
        --out    runs/dainet_full/checkpoints/model_final.pt
    # force live (non-EMA) weights instead:
    python scripts/finalize_checkpoint.py --latest … --out … --no_ema
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from training.callbacks import save_light_checkpoint  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--latest", required=True, help="Path to the full-state latest.pt.")
    p.add_argument("--out", required=True, help="Path to write the slim model_final.pt.")
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--use_ema", dest="use_ema", action="store_true", default=True,
                     help="Prefer EMA weights (default; matches the trainer).")
    grp.add_argument("--no_ema", dest="use_ema", action="store_false",
                     help="Use the live weights instead of the EMA shadow.")
    args = p.parse_args()

    latest_path = Path(args.latest)
    if not latest_path.exists():
        raise SystemExit(f"[finalize] not found: {latest_path}")

    ckpt = torch.load(str(latest_path), map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "model" in ckpt):
        raise SystemExit(
            f"[finalize] {latest_path} is not a full-state checkpoint "
            "(no 'model' key). Nothing to finalize."
        )
    cfg = ckpt.get("config", {})
    model_state = dict(ckpt["model"])  # full state dict (params + buffers)
    ema_state = ckpt.get("ema_state")

    source = "live weights"
    if args.use_ema and ema_state:
        # Overlay the EMA shadow (param tensors only) onto the full state dict,
        # leaving non-param buffers from the live model intact — exactly what
        # EMAModel.average_parameters(...) + model.state_dict() yields.
        overlaid = 0
        for k, v in ema_state.items():
            if k in model_state:
                model_state[k] = v
                overlaid += 1
        source = f"EMA weights ({overlaid} params overlaid)"
    elif args.use_ema and not ema_state:
        print("[finalize] WARNING: --use_ema set but no ema_state in checkpoint; "
              "falling back to live weights.")

    out_path = Path(args.out)
    save_light_checkpoint(model_state, cfg, out_path)
    size_mb = out_path.stat().st_size / 1e6
    epoch = ckpt.get("epoch")
    step = ckpt.get("step")
    print(
        f"[finalize] wrote {out_path} ({size_mb:.0f} MB) from {source} "
        f"(epoch={epoch}, step={step})."
    )


if __name__ == "__main__":
    main()
