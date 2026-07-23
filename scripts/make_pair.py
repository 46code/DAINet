"""dainet pair maker — one [input | prediction] PNG per input image.

Loads a checkpoint and writes a side-by-side before/after PNG for each input.
Each output file contains exactly **one** input and **one** prediction —
no combined grids. Use `scripts/infer.py` instead when you want only the
corrected image with no comparison.

Usage:
    python scripts/make_pair.py --ckpt checkpoints/model_best.pt \\
        --input photo.jpg --output corrected_pair.png

    python scripts/make_pair.py --ckpt checkpoints/model_best.pt \\
        --input_dir samples/ --output_dir out/pairs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

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
)


def _save_pair(input_u8: np.ndarray, pred_f: np.ndarray, out_path: Path) -> None:
    """Write a side-by-side [input | prediction] PNG.

    Both images share the same height (the input's native height); they are
    `np.hstack`-ed horizontally. No matplotlib — direct image write so the
    output is a clean image, not a matplotlib figure with axes/titles.
    """
    pred_u8 = np.clip(pred_f * 255.0, 0, 255).astype(np.uint8)
    if pred_u8.shape[:2] != input_u8.shape[:2]:
        pred_u8 = cv2.resize(
            pred_u8,
            (input_u8.shape[1], input_u8.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    pair = np.hstack([input_u8, pred_u8])
    bgr = cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), bgr)


def _pair_path(in_path: Path, out_path: Path) -> Path:
    """`foo.jpg` next to `foo.jpg` → `<out_dir>/foo_pair.png`."""
    return out_path.with_name(out_path.stem + "_pair.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--input", help="Single sRGB image path.")
    parser.add_argument(
        "--output",
        help="Single output pair path (will be normalised to '<stem>_pair.png').",
    )
    parser.add_argument("--input_dir", help="Directory of sRGB inputs.")
    parser.add_argument(
        "--output_dir",
        help="Directory for output pairs (one '<input>_pair.png' file per input).",
    )
    parser.add_argument(
        "--size",
        default="512,640",
        help="H,W resize for the network forward. Output is bicubic-upsampled "
             "back to the input's native resolution before pairing.",
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

    print(
        f"[dainet make_pair] device={device} ckpt={args.ckpt} "
        f"output_dir={args.output_dir or '<default>'} net_hw={net_hw}",
        flush=True,
    )

    model, _cfg = load_model_from_checkpoint(args.ckpt, device)
    log_priors("dainet make_pair", model, args)
    inputs = collect_inputs(args.input, args.output, args.input_dir, args.output_dir, default_out_dir="out/pairs")
    if not inputs:
        raise SystemExit(f"No input images found in {args.input_dir or args.input}.")
    print(f"[dainet make_pair] processing {len(inputs)} image(s)", flush=True)

    for index, (in_path, out_path) in enumerate(inputs, start=1):
        print(f"[dainet make_pair] [{index}/{len(inputs)}] {in_path}", flush=True)
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
        pair_out = _pair_path(in_path, out_path)
        _save_pair(img, pred, pair_out)
        print(
            f"[dainet make_pair] [{index}/{len(inputs)}] wrote {pair_out} "
            f"({2 * img.shape[1]}x{img.shape[0]})",
            flush=True,
        )


if __name__ == "__main__":
    main()
