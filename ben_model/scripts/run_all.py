"""End-to-end benchmark orchestrator: prep -> train -> infer -> score -> compare -> verify.

Each model runs as an isolated subprocess (so per-repo cwd / GPU pin / imports
never collide), and every stage records pass/fail per item and continues, so a
single model's failure does not abort the sweep. Defaults to the full sweep (all
985 scenes, native resolution); narrow it with ``--stages`` / ``--datasets`` /
``--limit_scenes`` / ``--max_side``.

Examples:
  python run_all.py --gpu 1                                       # full sweep
  python run_all.py --gpu 1 --stages prep,train
  python run_all.py --gpu 1 --stages infer,score,compare --datasets mit_mi
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
BASELINES = config.BASELINES                 # trained-from-scratch
ALL_MODELS = config.MODELS                   # baselines + dainet
ALL_STAGES = ["prep", "train", "infer", "score", "compare", "verify"]


def _run(cmd: list[str], log: list, tag: str, timeout: int) -> bool:
    print(f"\n>>> {tag}\n    $ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    try:
        subprocess.run(cmd, check=True, timeout=timeout)
        log.append({"tag": tag, "ok": True, "sec": round(time.time() - t0, 1)})
        return True
    except Exception as e:
        print(f"    !! FAILED: {e}", flush=True)
        log.append({"tag": tag, "ok": False, "sec": round(time.time() - t0, 1), "err": str(e)[:200]})
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--stages", default="all")
    ap.add_argument("--models", default=",".join(BASELINES))
    ap.add_argument("--datasets", default=",".join(config.DATASETS))
    ap.add_argument("--limit_scenes", type=int, default=0, help="0 = all scenes")
    ap.add_argument("--max_side", type=int, default=0, help="cap inference long side (0 = native)")
    ap.add_argument("--iters", type=int, default=0,
                    help="override BasicSR-trio train iters (0 = trainer default ~150k)")
    ap.add_argument("--resume", action="store_true",
                    help="resume each trainer (trio: step-exact `--resume auto`; natives: warm-start `--resume`)")
    ap.add_argument("--no_wandb", dest="wandb", action="store_false", default=True,
                    help="disable wandb logging in the trainers (default: log online)")
    ap.add_argument("--timeout", type=int, default=1209600)
    args = ap.parse_args()

    stages = ALL_STAGES if args.stages == "all" else args.stages.split(",")
    baselines = [m for m in args.models.split(",") if m in BASELINES]
    # by default, evaluate the trained baselines + dainet
    eval_models = baselines + (["dainet"] if "dainet" in args.models.split(",")
                               or args.models == ",".join(BASELINES) else [])
    datasets = [d for d in args.datasets.split(",") if d in config.DATASETS]
    py = config.PYTHON
    log: list = []

    if "prep" in stages:
        cmd = [py, str(SCRIPTS.parent / "data_prep" / "build_pairs.py"), "--scenes", "0"]
        _run(cmd, log, "prep:build_pairs", args.timeout)

    if "train" in stages:
        for m in baselines:
            trio = m in ("restormer", "retinexformer", "rln2")
            cmd = [py, str(SCRIPTS / "train" / f"train_{m}.py"), "--gpu", str(args.gpu)]
            if args.iters and trio:
                cmd += ["--iters", str(args.iters)]
            if args.resume:
                cmd += ["--resume", "auto"] if trio else ["--resume"]
            if not args.wandb:
                cmd += ["--no_wandb"]
            _run(cmd, log, f"train:{m}", args.timeout)

    if "infer" in stages:
        for d in datasets:
            for m in eval_models:
                cmd = [py, str(SCRIPTS / "methods" / f"run_{m}.py"), "--dataset", d,
                       "--gpu", str(args.gpu), "--limit_scenes", str(args.limit_scenes)]
                if m != "dainet":
                    cmd += ["--max_side", str(args.max_side)]
                _run(cmd, log, f"infer:{m}:{d}", args.timeout)

    if "score" in stages:
        for d in datasets:
            for m in eval_models:
                if not (config.RESULTS / d / m / "preds").exists():
                    continue
                _run([py, str(SCRIPTS / "eval_all.py"), "--dataset", d, "--model", m,
                      "--device", f"cuda:{args.gpu}", "--limit_scenes", str(args.limit_scenes)],
                     log, f"score:{m}:{d}", args.timeout)

    if "compare" in stages:
        for d in datasets:
            _run([py, str(SCRIPTS / "compare_all.py"), "--dataset", d], log, f"compare:{d}", args.timeout)

    status = {"ok": all(e["ok"] for e in log), "stages": stages, "log": log}
    (config.RESULTS).mkdir(parents=True, exist_ok=True)
    (config.RESULTS / "run_status.json").write_text(json.dumps(status, indent=2))

    print("\n================ RUN SUMMARY ================")
    for e in log:
        print(f"  [{'OK ' if e['ok'] else 'FAIL'}] {e['tag']}  ({e['sec']}s)"
              + ("" if e["ok"] else f"  -- {e.get('err','')}"))
    n_ok = sum(e["ok"] for e in log)
    print(f"  {n_ok}/{len(log)} steps OK  ->  results/run_status.json")

    if "verify" in stages:
        verify(datasets)


def verify(datasets: list[str]) -> None:
    print("\n================ VERIFY ================")
    problems = []
    for d in datasets:
        comp = config.RESULTS / d / "compare"
        for f in ("table1_mean_ci.csv", "table1_mean_ci.tex", "table1_mean_ci.png", "leaderboard.json"):
            if not (comp / f).exists():
                problems.append(f"{d}/compare/{f} missing")
        lb = comp / "leaderboard.json"
        if lb.exists():
            data = json.loads(lb.read_text())
            n = len(data.get("models", {}))
            print(f"  {d}: {n} models in leaderboard")
    if problems:
        print("  VERIFY problems:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  all expected compare artifacts present.")


if __name__ == "__main__":
    main()
