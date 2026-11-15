"""Report figures from recorded or reproduced results: accuracy across missingness conditions, training curves.

  severity-accuracy.pdf  top-1 annotation-match accuracy (primary metric, seeds pooled) of R, A0 and M in the
                         four conditions (none, low, mid, high); left panel validation, right panel held-out
                         test; error bars = the 95 % crossed-bootstrap intervals. Source: the "accuracy" entries
                         of h1.json (artifacts/records/<split>/h1.json, or a scripts/reproduce.py --out folder).
  training-curves.pdf    per epoch: training loss (total), validation top-1 (primary) without injected holes
                         and at mid severity; one thin line per seed, thick line = mean of the three seeds.
                         Source: the nine selected runs' epoch logs (artifacts/logs/<model>_s<seed>.jsonl).
No numbers are computed here beyond means of logged values; each figure gets a <name>.json next to it with the
plotted values and the sha256 of every input.

Usage:
  python scripts/make_paper_figures.py --out figures
  python scripts/make_paper_figures.py --h1-val repro_val/h1.json --h1-test repro_test/h1.json --out figures
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data.provenance import git_fields  # noqa: E402

ART = REPO / "artifacts"
MODELS, SEEDS, CONDITIONS = ("R", "A0", "M"), (1, 2, 3), ("none", "low", "mid", "high")
COLOR = {"R": "#009E73", "A0": "#0072B2", "M": "#E69F00"}
MARKER = {"R": "s", "A0": "o", "M": "^"}
SPLIT_NAME = {"val": "Validation", "test_module": "Held-out test"}


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def severity_figure(h1_paths, out, size, dpi):
    fig, axes = plt.subplots(1, 2, figsize=size, sharey=True)
    plotted, offsets = {}, {"R": -0.12, "A0": 0.0, "M": 0.12}
    for ax, (split, path) in zip(axes, h1_paths.items()):
        res = json.loads(Path(path).read_text())
        if res.get("split") != split:
            raise SystemExit(f"{path}: split {res.get('split')!r}, expected {split!r}")
        if res.get("metric") != "primary":
            raise SystemExit(f"{path}: metric {res.get('metric')!r}; the figure shows the primary metric")
        plotted[split] = {}
        x = np.arange(len(CONDITIONS))
        for m in MODELS:
            acc = [res["accuracy"][f"{m} {c}"] for c in CONDITIONS]
            y = np.array([v["acc_a_pct"] for v in acc])
            lo = y - np.array([v["ci95"][0] for v in acc])
            hi = np.array([v["ci95"][1] for v in acc]) - y
            ax.errorbar(x + offsets[m], y, yerr=[lo, hi], color=COLOR[m], marker=MARKER[m], ms=4, lw=1.2,
                        capsize=2, label=m)
            plotted[split][m] = {c: {"acc_pct": v["acc_a_pct"], "ci95": v["ci95"]} for c, v in zip(CONDITIONS, acc)}
        ax.set_xticks(x, CONDITIONS)
        ax.set_xlabel("synthetic missingness")
        ax.set_title(SPLIT_NAME[split], fontsize=9)
        ax.grid(axis="y", lw=0.4, alpha=0.5)
    axes[0].set_ylabel("top-1 accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return {"inputs": {s: {"path": Path(p).name, "sha256": sha256_file(p)} for s, p in h1_paths.items()},
            "values": plotted}


def training_figure(log_dir, out, size, dpi):
    logs = {}
    for m in MODELS:
        for s in SEEDS:
            p = Path(log_dir) / f"{m}_s{s}.jsonl"
            recs = sorted((json.loads(x) for x in p.read_text().splitlines() if x.strip()), key=lambda r: r["epoch"])
            if [r["epoch"] for r in recs] != list(range(len(recs))):
                raise SystemExit(f"{p}: epochs are not 0..{len(recs) - 1}")
            logs[(m, s)] = recs
    panels = [("training loss (total)", lambda r: r["loss"]["total"]),
              ("val top-1 (%), no injected holes", lambda r: 100.0 * r["val"]["none"]["top1_primary"]),
              ("val top-1 (%), mid severity", lambda r: 100.0 * r["val"]["mid"]["top1_primary"])]
    fig, axes = plt.subplots(1, len(panels), figsize=size)
    plotted = {}
    for ax, (name, f) in zip(axes, panels):
        plotted[name] = {}
        for m in MODELS:
            ys = np.array([[f(r) for r in logs[(m, s)]] for s in SEEDS])
            ep = np.arange(1, ys.shape[1] + 1)
            for y in ys:
                ax.plot(ep, y, color=COLOR[m], lw=0.6, alpha=0.35)
            ax.plot(ep, ys.mean(0), color=COLOR[m], lw=1.6, label=m)
            plotted[name][m] = {"per_seed": ys.tolist(), "mean": ys.mean(0).tolist()}
        ax.set_xlabel("epoch")
        ax.set_title(name, fontsize=8)
        ax.grid(lw=0.4, alpha=0.5)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return {"inputs": {f"{m}_s{s}": sha256_file(Path(log_dir) / f"{m}_s{s}.jsonl") for m in MODELS for s in SEEDS},
            "values": plotted}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h1-val", default=str(ART / "records" / "val" / "h1.json"))
    p.add_argument("--h1-test", default=str(ART / "records" / "test_module" / "h1.json"))
    p.add_argument("--logs", default=str(ART / "logs"), help="folder with <model>_s<seed>.jsonl")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--format", default="pdf", choices=("pdf", "png"))
    p.add_argument("--width-in", type=float, default=7.0)
    p.add_argument("--height-in", type=float, default=2.4)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--seed", type=int, default=0, help="unused (no randomness); recorded by convention")
    return p.parse_args()


def main():
    a = parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    size = (a.width_in, a.height_in)
    jobs = {"severity-accuracy": lambda p: severity_figure({"val": a.h1_val, "test_module": a.h1_test}, p, size, a.dpi),
            "training-curves": lambda p: training_figure(a.logs, p, size, a.dpi)}
    for name, job in jobs.items():
        fig_path = out / f"{name}.{a.format}"
        cfg_path = out / f"{name}.json"
        if not a.overwrite and (fig_path.exists() or cfg_path.exists()):
            raise SystemExit(f"{fig_path} or {cfg_path} exists; pass --overwrite")
        rec = job(fig_path)
        rec.update(script="scripts/make_paper_figures.py", figure=fig_path.name,
                   args={k: v for k, v in vars(a).items() if k not in ("h1_val", "h1_test", "logs", "out")},
                   **git_fields())
        cfg_path.write_text(json.dumps(rec, indent=1) + "\n")
        print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
