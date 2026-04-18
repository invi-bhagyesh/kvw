"""
Stage A analysis: does residual redundancy r_i predict forget-sample
recoverability after KVW unlearning?

Inputs:
  --r-path         : kc/r_stageA_forget05.pt   (from KVW.py --phase stage_a_measure)
  --eval-path      : <output_folder>/forget_per_sample.json   (from eval.py)
  --output-dir     : where to write plot + summary

Outputs:
  auc.json              : {"auc": float, "n": int, "n_correct": int}
  stage_a_scatter.png   : r_i vs correctness (jittered)
  stage_a_bins.png      : quintile bins, P(correct) per bin
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def _normalize_name(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def load_r(r_path):
    blob = torch.load(r_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or "r" not in blob or "names" not in blob:
        raise ValueError(
            f"{r_path} does not contain {{r, names}} — rerun stage_a_measure "
            "after the name-join patch."
        )
    return blob["r"].numpy(), [_normalize_name(n) for n in blob["names"]]


def load_correct(eval_path):
    with open(eval_path) as f:
        rows = json.load(f)
    return {_normalize_name(r["name"]): int(r["correct"]) for r in rows}


def join_by_name(r, r_names, correct_by_name):
    paired_r, paired_c, unmatched = [], [], []
    for score, name in zip(r, r_names):
        if name in correct_by_name:
            paired_r.append(score)
            paired_c.append(correct_by_name[name])
        else:
            unmatched.append(name)
    return np.array(paired_r), np.array(paired_c, dtype=np.int32), unmatched


def compute_auc(scores, labels):
    # Manual AUC so we don't require sklearn.
    # AUC = P(score_pos > score_neg)
    order = np.argsort(-scores)  # descending
    labels = labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    cum_neg = 0
    tp_over_fp = 0
    for y in labels:
        if y == 0:
            cum_neg += 1
        else:
            tp_over_fp += cum_neg
    return tp_over_fp / (n_pos * n_neg)


def quintile_bins(scores, labels, n_bins=5):
    order = np.argsort(scores)
    scores = scores[order]
    labels = labels[order]
    edges = np.linspace(0, len(scores), n_bins + 1, dtype=int)
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if hi <= lo:
            continue
        bin_scores = scores[lo:hi]
        bin_labels = labels[lo:hi]
        rows.append({
            "bin": b,
            "n": int(hi - lo),
            "r_min": float(bin_scores.min()),
            "r_max": float(bin_scores.max()),
            "p_correct": float(bin_labels.mean()),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-path", required=True)
    ap.add_argument("--eval-path", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    r, r_names = load_r(args.r_path)
    correct_by_name = load_correct(args.eval_path)

    r, correct, unmatched = join_by_name(r, r_names, correct_by_name)
    if len(r) == 0:
        raise ValueError(
            "No names matched between r and eval output. "
            "Check that the forget split used by KVW and eval.py share identities."
        )
    if unmatched:
        print(f"[warn] {len(unmatched)} r samples had no matching eval entry "
              f"(showing up to 5): {unmatched[:5]}")

    auc = compute_auc(r.astype(np.float64), correct)
    bins = quintile_bins(r, correct)

    summary = {
        "n": int(len(r)),
        "n_unmatched": int(len(unmatched)),
        "n_correct": int(correct.sum()),
        "p_correct_overall": float(correct.mean()),
        "auc": float(auc),
        "r_mean": float(r.mean()),
        "r_std": float(r.std()),
        "bins": bins,
    }

    out_file = os.path.join(args.output_dir, "auc.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Summary saved to {out_file}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.04, 0.04, size=len(correct))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(r, correct + jitter, alpha=0.5, s=12)
    ax.set_xlabel("r_i  (residual redundancy after KVW)")
    ax.set_ylabel("still correct on forget eval (0/1)")
    ax.set_title(f"Stage A — AUC = {auc:.3f}  (n={len(r)})")
    ax.set_yticks([0, 1])
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "stage_a_scatter.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(bins))
    ys = [b["p_correct"] for b in bins]
    ax.bar(xs, ys, color="steelblue")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{b['bin']+1}\n[{b['r_min']:.1f},{b['r_max']:.1f}]"
                        for b in bins], fontsize=8)
    ax.set_ylabel("P(still correct)")
    ax.set_ylim(0, 1)
    ax.set_title(f"Stage A — quintile bins of r_i  (AUC={auc:.3f})")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "stage_a_bins.png"), dpi=140)
    print(f"Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
