"""dainet inference — single corrected image per input, at the input's native resolution.

Loads a checkpoint and writes one corrected image for each input. SAM2 runs
inline at the network resolution; the network output is then bicubic-upsampled
back to the input's original (H, W) so saved files match the input dimensions.

This script intentionally produces **one** output image per input. For a
side-by-side before/after pair (one PNG per input), use `scripts/make_pair.py`.

Usage:
    python scripts/infer.py --ckpt checkpoints/model_best.pt \\
        --input photo.jpg --output corrected.png

    python scripts/infer.py --ckpt checkpoints/model_best.pt \\
        --input_dir my_scene/ --output_dir corrected/

    python scripts/infer.py --ckpt checkpoints/model_best.pt \\
        --input_dir data/raw/online --output_dir out/online \\
        --sam_weights ~/my_model/sam2_weights/sam2.1_hiera_large.pt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Disable HuggingFace hub network access to prevent model download hangs.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TIMM_HOME", os.path.expanduser("~/.cache/timm"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._infer_common import (  # noqa: E402
    add_prior_args,
    collect_inputs,
    infer_single,
    load_image_rgb,
    load_model_from_checkpoint,
    log_priors,
    parse_size_arg,
    resolve_device,
    save_image_rgb,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", help="Single sRGB image path.")
    parser.add_argument("--output", help="Single output path.")
    parser.add_argument("--input_dir", help="Directory of sRGB inputs.")
    parser.add_argument("--output_dir", help="Directory for outputs.")
    parser.add_argument(
        "--size",
        default="512,640",
        help="H,W resize for the network forward. Output is bicubic-upsampled "
             "back to the input's native resolution before saving.",
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument(
        "--sam_weights",
        default="~/my_model/sam2_weights/sam2.1_hiera_large.pt",
        help="path to SAM2 checkpoint weights",
    )
    add_prior_args(parser)
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        raise SystemExit("Need --input or --input_dir.")

    device = resolve_device(args.gpu)
    net_hw = parse_size_arg(args.size)

    source_desc = f"input={args.input}" if args.input else f"input_dir={args.input_dir}"
    print(
        f"[dainet infer] device={device} ckpt={args.ckpt} {source_desc} "
        f"output_dir={args.output_dir or '<default>'} net_hw={net_hw}",
        flush=True,
    )

    print("[dainet infer] loading checkpoint and model...", flush=True)
    model, _cfg = load_model_from_checkpoint(args.ckpt, device)
    print("[dainet infer] model ready", flush=True)
    log_priors("dainet infer", model, args)

    inputs = collect_inputs(args.input, args.output, args.input_dir, args.output_dir)
    if not inputs:
        raise SystemExit(f"No input images found in {args.input_dir or args.input}.")
    print(f"[dainet infer] processing {len(inputs)} image(s)", flush=True)

    for index, (in_path, out_path) in enumerate(inputs, start=1):
        print(f"[dainet infer] [{index}/{len(inputs)}] {in_path}", flush=True)
        img = load_image_rgb(in_path)
        pred = infer_single(
            model,
            img,
            net_hw=net_hw,
            device=device,
            sam_weights=args.sam_weights,
            compute_normals=not args.no_normals,
            estimate_illuminant=args.estimate_illuminant,
            illuminant_strength=args.illuminant_strength,
        )
        save_image_rgb(pred, out_path)
        print(
            f"[dainet infer] [{index}/{len(inputs)}] wrote {out_path} "
            f"({pred.shape[1]}x{pred.shape[0]})",
            flush=True,
        )


if __name__ == "__main__":
    main()
