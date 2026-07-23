"""Aggregate per-model results on a dataset into publication outputs.

Auto-discovers every ``results/<dataset>/<model>/metrics.json`` and emits, under
``results/<dataset>/compare/``:

  table1_mean_ci.{csv,tex,png}   headline mean (+-95% bootstrap CI) per metric;
                                 best per column bold, dainet row bold, Holm-adjusted
                                 Wilcoxon vs dainet marked * (p<.05) / ** (p<.01).
  stats_significance.csv         Delta-vs-dainet, bootstrap CI, Wilcoxon p, Holm p, Cohen's d.
  bar_<metric>.png               per-metric bar chart across models.
  qualitative.png                Input | <each model> | GT strip (a few samples).
  leaderboard.json, SUMMARY.md   machine-readable + human summary.

The statistical apparatus (paired bootstrap CI, Wilcoxon signed-rank, Holm-
Bonferroni over the model x metric family, Cohen's d) follows ben_guide.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Larger, consistent type: these figures are shrunk to \linewidth in the thesis.
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
})

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config, datasets  # noqa: E402
import stats as st  # noqa: E402

# metric -> (display, higher_is_better, fmt)
METRICS = {
    "psnr": ("PSNR↑", True, "{:.2f}"),
    "ms_ssim": ("MS-SSIM↑", True, "{:.4f}"),
    "lpips": ("LPIPS↓", False, "{:.4f}"),
}
TEX_NAME = {
    "psnr": r"PSNR$\uparrow$", "ms_ssim": r"MS-SSIM$\uparrow$", "lpips": r"LPIPS$\downarrow$",
}
MODEL_LABEL = {"restormer": "Restormer", "retinexformer": "Retinexformer", "rln2": "RLN2",
               "ifblend": "IFBlend", "dainet": "DAINet (ours)"}
ORDER = ["restormer", "retinexformer", "rln2", "ifblend", "dainet"]


def load_results(dataset: str, results_root: Path) -> dict:
    root = results_root / dataset
    out = {}
    for mj in sorted(root.glob("*/metrics.json")):
        model = mj.parent.name
        if model == "compare":
            continue
        data = json.loads(mj.read_text())
        per = {r["key"]: r for r in data.get("per_sample", [])}
        rec = {"agg": data.get("aggregates", {}), "per_sample": per}
        out[model] = rec
    return out


def _models_sorted(res: dict) -> list[str]:
    return [m for m in ORDER if m in res] + [m for m in res if m not in ORDER]


def _paired(res: dict, a: str, b: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Aligned per-sample values for two models on a metric (shared keys)."""
    keys = sorted(set(res[a]["per_sample"]) & set(res[b]["per_sample"]))
    va, vb = [], []
    for k in keys:
        x, y = res[a]["per_sample"][k].get(metric), res[b]["per_sample"][k].get(metric)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            va.append(x); vb.append(y)
    return np.array(va), np.array(vb)


def compute_significance(res: dict, metric_keys: list[str]) -> dict:
    """Holm-adjusted Wilcoxon + bootstrap CI + Cohen's d for each model vs dainet."""
    if "dainet" not in res:
        return {}
    pairs, raw = [], []
    for m in res:
        if m == "dainet":
            continue
        for metric in metric_keys:
            a, b = _paired(res, m, "dainet", metric)  # model vs dainet
            if len(a) >= 3:
                r = st.compare_paired(a, b, n_boot=2000)
                pairs.append((m, metric)); raw.append(r)
    if not raw:
        return {}
    adj = st.holm_bonferroni([r.wilcoxon_p if np.isfinite(r.wilcoxon_p) else 1.0 for r in raw])
    sig = {}
    for (m, metric), r, p in zip(pairs, raw, adj):
        sig.setdefault(m, {})[metric] = {
            "delta_vs_dainet": r.delta_mean, "ci_lo": r.ci_lo, "ci_hi": r.ci_hi,
            "wilcoxon_p": r.wilcoxon_p, "holm_p": p, "cohens_d": r.cohens_d, "n": r.n}
    return sig


def _best_per_metric(res: dict, models: list[str], keys: list[str]) -> dict:
    best = {}
    for k in keys:
        vals = {m: res[m]["agg"].get(k, {}).get("mean") for m in models
                if res[m]["agg"].get(k, {}).get("mean") is not None}
        if not vals:
            continue
        hib = METRICS.get(k, (None, False))[1] if k in METRICS else False
        best[k] = (max if hib else min)(vals, key=vals.get)
    return best


def write_table1(dataset: str, res: dict, sig: dict, out: Path) -> list[str]:
    models = _models_sorted(res)
    keys = [k for k in METRICS if any(res[m]["agg"].get(k) for m in models)]
    best = _best_per_metric(res, models, keys)
    # CSV
    rows = []
    for m in models:
        row = {"model": m}
        for k in keys:
            a = res[m]["agg"].get(k, {})
            row[k] = a.get("mean")
            row[f"{k}_ci95"] = 1.96 * a.get("std", 0) / max(a.get("count", 1) ** 0.5, 1)
        rows.append(row)
    cols = ["model"] + sum([[k, f"{k}_ci95"] for k in keys], [])
    with (out / "table1_mean_ci.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # LaTeX (booktabs)
    disp_keys = keys
    lines = [r"\begin{tabular}{l" + "c" * len(disp_keys) + "}", r"\toprule",
             "Method & " + " & ".join(TEX_NAME.get(k, k) for k in disp_keys) + r" \\", r"\midrule"]
    for m in models:
        cells = []
        for k in disp_keys:
            v = res[m]["agg"].get(k, {}).get("mean")
            if v is None:
                cells.append("--"); continue
            fmt = METRICS[k][2] if k in METRICS else "{:.4f}"
            s = fmt.format(v)
            if best.get(k) == m:
                s = r"\textbf{" + s + "}"
            star = ""
            if m in sig and k in sig.get(m, {}):
                p = sig[m][k]["holm_p"]
                star = "$^{**}$" if p < 0.01 else ("$^{*}$" if p < 0.05 else "")
            cells.append(s + star)
        name = MODEL_LABEL.get(m, m)
        if m == "dainet":
            name = r"\textbf{" + name + "}"
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "table1_mean_ci.tex").write_text("\n".join(lines))

    # PNG (matplotlib table)
    disp = [MODEL_LABEL.get(m, m) for m in models]
    header = [METRICS[k][0] if k in METRICS else k for k in disp_keys]
    cell_text = []
    for m in models:
        rr = []
        for k in disp_keys:
            v = res[m]["agg"].get(k, {}).get("mean")
            fmt = METRICS[k][2] if k in METRICS else "{:.4f}"
            rr.append("--" if v is None else fmt.format(v))
        cell_text.append(rr)
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(disp_keys), 0.5 + 0.4 * len(models)))
    ax.axis("off")
    tbl = ax.table(cellText=cell_text, rowLabels=disp, colLabels=header, loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    fig.savefig(out / "table1_mean_ci.png", dpi=300, bbox_inches="tight"); plt.close(fig)
    return disp_keys


def write_stats(res: dict, sig: dict, out: Path) -> None:
    if not sig:
        return
    with (out / "stats_significance.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "metric", "delta_vs_dainet", "ci_lo", "ci_hi",
                    "wilcoxon_p", "holm_p", "cohens_d", "n"])
        for m, md in sig.items():
            for metric, s in md.items():
                w.writerow([m, metric, s["delta_vs_dainet"], s["ci_lo"], s["ci_hi"],
                            s["wilcoxon_p"], s["holm_p"], s["cohens_d"], s["n"]])


def bar_charts(res: dict, keys: list[str], out: Path) -> None:
    models = _models_sorted(res)
    for k in keys:
        vals = [(MODEL_LABEL.get(m, m), res[m]["agg"].get(k, {}).get("mean")) for m in models]
        vals = [(n, v) for n, v in vals if v is not None]
        if not vals:
            continue
        names, ys = zip(*vals)
        fig, ax = plt.subplots(figsize=(1.2 + 0.8 * len(names), 3))
        colors = ["#c44" if n.startswith("dainet") else "#48a" for n in names]
        ax.bar(names, ys, color=colors)
        ax.set_ylabel(METRICS[k][0] if k in METRICS else k)
        ax.set_title(k); plt.xticks(rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(out / f"bar_{k}.png", dpi=200); plt.close(fig)


def _distinct_scene_samples(dataset: str, n: int):
    """First sample of each of the first ``n`` distinct scenes, so the montage
    shows n *different* scenes (bigger, more informative) rather than n
    directions of one scene."""
    out, seen = [], set()
    for s in datasets.iter_samples(dataset, limit_scenes=0):
        if s.scene in seen:
            continue
        seen.add(s.scene)
        out.append(s)
        if len(out) >= n:
            break
    return out


def _cap(im, side=640):
    """Downscale so the longest side is <= `side`px: keeps the montage fast to
    render and the PNG small even when a dataset ships 4K frames (cl3an)."""
    import cv2
    if im is None:
        return None
    h, w = im.shape[:2]
    m = max(h, w)
    if m > side:
        im = cv2.resize(im, (max(1, int(w * side / m)), max(1, int(h * side / m))),
                        interpolation=cv2.INTER_AREA)
    return im


def qualitative(dataset: str, res: dict, out: Path, n_samples: int = 2,
                out_name: str = "qualitative", bands: bool = True) -> None:
    """Qualitative comparison montage.

    ``bands=True`` (headline figures): the columns (Input + each model + GT) are
    wrapped into TWO stacked bands per scene, so each panel is drawn ~2x larger
    than a single wide row would allow; every band-row carries its own column
    titles, so all panels are labelled. Best for 1--2 scenes.

    ``bands=False`` (appendix galleries): one wide row per scene (Input | models
    | GT), which stays short enough to stack many scenes on a single page."""
    import cv2  # noqa: F401
    models = _models_sorted(res)
    samples = _distinct_scene_samples(dataset, n_samples)
    if not samples or not models:
        return
    cols = ["Input"] + [MODEL_LABEL.get(m, m) for m in models] + ["GT"]
    L = len(cols)

    def _panels(s):
        p = [_cap(datasets.read_rgb01(s.input_path))]
        for m in models:
            pp = config.RESULTS / dataset / m / "preds" / f"{s.key}.png"
            p.append(_cap(datasets.read_rgb01(pp)) if pp.exists() else None)
        p.append(_cap(datasets.read_rgb01(s.gt_path)))
        return p

    if bands:
        per = (L + 1) // 2                   # columns per band -> 2 bands
        nb = 2
        nrow = len(samples) * nb
        fig, axes = plt.subplots(nrow, per, figsize=(3.1 * per, 3.1 * nrow),
                                 squeeze=False)
        for r in range(nrow):
            for c in range(per):
                axes[r][c].axis("off")
        for i, s in enumerate(samples):
            panels = _panels(s)
            for b in range(nb):
                base = b * per
                r = i * nb + b
                for c in range(per):
                    idx = base + c
                    if idx >= L:
                        continue
                    if panels[idx] is not None:
                        axes[r][c].imshow(np.clip(panels[idx], 0, 1))
                    axes[r][c].set_title(cols[idx], fontsize=16, fontweight="bold")
    else:
        fig, axes = plt.subplots(len(samples), L, figsize=(2.4 * L, 2.4 * len(samples)),
                                 squeeze=False)
        for i, s in enumerate(samples):
            panels = _panels(s)
            for j, im in enumerate(panels):
                ax = axes[i][j]
                ax.axis("off")
                if im is not None:
                    ax.imshow(np.clip(im, 0, 1))
                if i == 0:
                    ax.set_title(cols[j], fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out / f"{out_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(dataset: str, res: dict, sig: dict, keys: list[str], out: Path) -> None:
    models = _models_sorted(res)
    lb = {"dataset": dataset, "models": {}}
    for m in models:
        lb["models"][m] = {k: res[m]["agg"].get(k, {}).get("mean") for k in METRICS
                           if res[m]["agg"].get(k)}
    (out / "leaderboard.json").write_text(json.dumps(lb, indent=2))
    md = [f"# Benchmark summary — {dataset}", "",
          f"Models: {', '.join(MODEL_LABEL.get(m, m) for m in models)}", "",
          "See `table1_mean_ci.{csv,tex,png}` for the headline table"
          + (", `stats_significance.csv` for Wilcoxon/Holm vs dainet." if sig else "."), ""]
    if "dainet" in res:
        md.append("`*`/`**` mark Holm-adjusted Wilcoxon p<.05/.01 (dainet vs baseline).")
    (out / "SUMMARY.md").write_text("\n".join(md))


def run(dataset: str, results_root: Path | None = None) -> None:
    results_root = results_root or config.RESULTS
    res = load_results(dataset, results_root)
    if not res:
        print(f"[compare] no results for {dataset}"); return
    out = results_root / dataset / "compare"
    out.mkdir(parents=True, exist_ok=True)
    metric_keys = [k for k in METRICS if any(res[m]["agg"].get(k) for m in res)]
    sig = compute_significance(res, metric_keys)
    disp_keys = write_table1(dataset, res, sig, out)
    write_stats(res, sig, out)
    bar_charts(res, disp_keys, out)
    try:
        qualitative(dataset, res, out, n_samples=1)
        qualitative(dataset, res, out, n_samples=6,
                    out_name="qualitative_gallery", bands=False)
    except Exception as e:  # qualitative is best-effort
        print(f"[compare] qualitative skipped: {e}")
    write_summary(dataset, res, sig, disp_keys, out)
    print(f"[compare] {dataset}: {len(res)} models -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="dataset name or 'all'")
    args = ap.parse_args()
    ds = list(config.DATASETS) if args.dataset == "all" else [args.dataset]
    for d in ds:
        run(d)


if __name__ == "__main__":
    main()
