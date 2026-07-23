"""Train Retinexformer (ICCV'23) from scratch on MIT-MI with the unified L1 trainer.

Retinexformer's LOL config uses a pure L1 pixel loss; the shared trainer
reproduces it. We use the canonical single-stage recipe (n_feat=40, stage=1,
num_blocks=[1,2,2]) from ``Options/RetinexFormer_LOL_v1.yml``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config            # noqa: E402
from train import _common         # noqa: E402
from train._simple_trainer import load_arch_class, train_model  # noqa: E402

MODEL = "retinexformer"


def main() -> None:
    args = _common.base_parser(MODEL, default_patch=128).parse_args()
    r = _common.resolve(args, MODEL)
    RetinexFormer = load_arch_class(config.repo("retinexformer"),
                                    "basicsr/models/archs/RetinexFormer_arch.py", "RetinexFormer")
    net = RetinexFormer(in_channels=3, out_channels=3, n_feat=40, stage=1,
                        num_blocks=[1, 2, 2])
    print(f"[train_retinexformer] iters={r['iters']} batch={r['batch']} "
          f"patch={r['patch']} -> {r['out_dir']}")
    arch_kwargs = {"n_feat": 40, "stage": 1, "num_blocks": [1, 2, 2]}
    cfg = {"model": MODEL, "trainer": "unified_l1", "iters": r["iters"], "batch": r["batch"],
           "patch": r["patch"], "lr": r["lr"], "arch": arch_kwargs}
    wb = _common.make_wandb(args, MODEL, cfg, r["out_dir"])
    train_model(net, r["data_root"], r["out_dir"], iters=r["iters"], batch=r["batch"],
                patch=r["patch"], lr=r["lr"], device=r["device"],
                num_workers=r["num_workers"], resume=r["resume"],
                ckpt_every=r["ckpt_every"], val_every=r["val_every"],
                val_max_side=r["val_max_side"],
                model_name=MODEL, arch_kwargs=arch_kwargs, wandb_run=wb)


if __name__ == "__main__":
    main()
