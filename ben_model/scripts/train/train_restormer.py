"""Train Restormer (CVPR'22) from scratch on MIT-MI with the unified L1 trainer.

Restormer's restoration configs use a pure L1 pixel loss, which the shared
trainer reproduces. Architecture imported from the cloned repo
(``repos/restormer/basicsr/models/archs/restormer_arch.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config            # noqa: E402
from train import _common         # noqa: E402
from train._simple_trainer import load_arch_class, train_model  # noqa: E402

MODEL = "restormer"


def main() -> None:
    args = _common.base_parser(MODEL, default_patch=128).parse_args()
    r = _common.resolve(args, MODEL)
    Restormer = load_arch_class(config.repo("restormer"),
                                "basicsr/models/archs/restormer_arch.py", "Restormer")
    # Restormer default restoration config (dim=48, 4/6/6/8 blocks).
    net = Restormer(inp_channels=3, out_channels=3, dim=48,
                    num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                    heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False,
                    LayerNorm_type="WithBias")
    print(f"[train_restormer] iters={r['iters']} batch={r['batch']} "
          f"patch={r['patch']} -> {r['out_dir']}")
    arch_kwargs = {"dim": 48, "num_blocks": [4, 6, 6, 8], "heads": [1, 2, 4, 8]}
    # Restormer's official restoration configs train in fp32 (no `use_amp`),
    # unlike Retinexformer/RLN2 (use_amp: True) — keep its original precision.
    cfg = {"model": MODEL, "trainer": "unified_l1", "iters": r["iters"], "batch": r["batch"],
           "patch": r["patch"], "lr": r["lr"], "amp": False, "arch": arch_kwargs}
    wb = _common.make_wandb(args, MODEL, cfg, r["out_dir"])
    train_model(net, r["data_root"], r["out_dir"], iters=r["iters"], batch=r["batch"],
                patch=r["patch"], lr=r["lr"], device=r["device"], amp=False,
                num_workers=r["num_workers"], resume=r["resume"],
                ckpt_every=r["ckpt_every"], val_every=r["val_every"],
                val_max_side=r["val_max_side"],
                model_name=MODEL, arch_kwargs=arch_kwargs, wandb_run=wb)


if __name__ == "__main__":
    main()
