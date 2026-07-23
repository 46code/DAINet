"""Train RLN2 (ICCV'25, "After the Party") from scratch on MIT-MI.

RLN2 ships only its CC36 architecture + a scene dataset on top of a stripped
BasicSR (no train.py / model class), and its README directs you to the IFBlend
pipeline for training. Its config (Options/RLN2-Lf.yml) uses a pure L1 pixel
loss, so we train the CC36 architecture with the shared unified L1 trainer —
faithful to RLN2's objective and fair vs. the other BasicSR baselines.

CC36 builds a ConvNeXt-XL encoder that loads the generic ImageNet backbone from
``./weights/convnext_xlarge_22k_1k_384_ema.pth`` (allowed: generic backbone,
not a task checkpoint). We symlink the cached backbone into the repo and run
from the repo dir so that relative load succeeds. RLN2-Lf => num_blocks=4.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config            # noqa: E402
from train import _common         # noqa: E402
from train._simple_trainer import load_arch_class, train_model  # noqa: E402

MODEL = "rln2"


def main() -> None:
    ap = _common.base_parser(MODEL, default_patch=256)
    ap.add_argument("--num_blocks", type=int, default=4, help="RLN2-Lf => 4")
    args = ap.parse_args()
    r = _common.resolve(args, MODEL)
    batch = r["batch"]

    repo = config.repo("rln2")
    # Make the generic ImageNet ConvNeXt-XL backbone available at ./weights/...
    (repo / "weights").mkdir(exist_ok=True)
    link = repo / "weights" / "convnext_xlarge_22k_1k_384_ema.pth"
    if not link.exists():
        link.symlink_to((config.BACKBONES / "convnext_xlarge_22k_1k_384_ema.pth").resolve())
    os.chdir(repo)  # CC36.__init__ does torch.load('./weights/...')

    CC36 = load_arch_class(repo, "basicsr/models/archs/cc36_arch.py", "CC36")
    net = CC36(num_feats=16, num_blocks=args.num_blocks)
    print(f"[train_rln2] iters={r['iters']} batch={batch} "
          f"patch={r['patch']} num_blocks={args.num_blocks} -> {r['out_dir']}")
    arch_kwargs = {"num_feats": 16, "num_blocks": args.num_blocks}
    cfg = {"model": MODEL, "trainer": "unified_l1", "iters": r["iters"], "batch": batch,
           "patch": r["patch"], "lr": r["lr"], "arch": arch_kwargs}
    wb = _common.make_wandb(args, MODEL, cfg, r["out_dir"])
    train_model(net, r["data_root"].resolve(), r["out_dir"].resolve(),
                iters=r["iters"], batch=batch, patch=r["patch"], lr=r["lr"],
                device=r["device"], num_workers=r["num_workers"],
                resume=r["resume"], ckpt_every=r["ckpt_every"], val_every=r["val_every"],
                val_max_side=r["val_max_side"],
                model_name=MODEL, arch_kwargs=arch_kwargs, wandb_run=wb)


if __name__ == "__main__":
    main()
