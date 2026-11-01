#!/usr/bin/env python
# Laura Belli, October 2026
# Script to plot PSC timecourse of n ROIs with highest activation peak/mean over trial

"""
Timecourses of the N ROIs (default 5) with the highest PSC, per subject.

One figure per subject, one column per condition (16, 40, 80, 160 bins).
By default the ROIs are chosen once per subject in a reference condition (160 bins)
and the same ROIs/colours are shown in all four panels, so conditions are directly comparable.
Use --select per-condition to pick the top ROIs separately in each panel.

Output: <out>/sub-XX_top<N>_timecourses.<fmt>

Usage:
  python 5.1plot_top_timecourses.py --deriv ../derivatives --out ../derivatives/figures_top5 \
    --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 --n 5 --rank mean --select reference --ref 40bins --fmt png \
    --labels ../derivatives/templates/Schaefer2018_200Parcels_7Networks_order.txt

Options:
  --rank peak|mean       rank ROIs by max PSC (default) or mean PSC over the window
  --visual-only          restrict the candidates to visual-network ROIs (needs --labels)
  --zscore               plot z-scored time courses instead of raw PSC
"""
import argparse
import io
import re
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONDITIONS = [  # (label, n bins, TR in seconds)
    ("16bins", 16, 2.5),
    ("40bins", 40, 1.0),
    ("80bins", 80, 0.5),
    ("160bins", 160, 0.25),
]
PSC_PATTERN = "{deriv}/sub-{sub:02d}/reg/sub-{sub:02d}_task-CS_acq-{acq}bins_200Schaefer_psc.txt"
VIS_REGEX = re.compile(r"(^|_)Vis")

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def load_psc(deriv, sub, acq):
    path = Path(PSC_PATTERN.format(deriv=deriv, sub=sub, acq=acq))
    if not path.exists():
        print(f"[skip] missing {path}")
        return None
    return np.genfromtxt(path)  # time x ROI


def zscore_cols(x):
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x, axis=0)
    sd[sd == 0] = np.nan
    return (x - mu) / sd


def load_labels(path, n_roi=200):
    if path and Path(path).exists():
        labs = []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                name = parts[1] if len(parts) > 1 and parts[0].isdigit() else parts[0]
                labs.append(name.replace("7Networks_", "").replace("17Networks_", ""))
        if len(labs) >= n_roi:
            return labs[:n_roi]
        print(f"[warn] label file has {len(labs)} entries (< {n_roi}); ignoring it")
    return [f"ROI {i + 1}" for i in range(n_roi)]


def top_rois(x, n, rank, candidates):
    """Indices (into the 200 ROIs) of the n best ROIs among `candidates`."""
    score = np.nanmax(x, axis=0) if rank == "peak" else np.nanmean(x, axis=0)
    score = np.where(np.isnan(score), -np.inf, score)
    cand = np.asarray(candidates)
    return cand[np.argsort(score[cand])[::-1][:n]]


def plot_subject(deriv, sub, out, fmt, n, rank, select, ref, labels, candidates, zscore):
    data = {}
    for cond, acq, tr in CONDITIONS:
        x = load_psc(deriv, sub, acq)
        if x is not None:
            data[cond] = zscore_cols(x) if zscore else x
    if not data:
        return

    shared = None
    if select == "reference":
        ref_cond = ref if ref in data else next(iter(data))
        if ref_cond != ref:
            print(f"[warn] reference {ref} missing for sub-{sub:02d}; using {ref_cond}")
        shared = top_rois(data[ref_cond], n, rank, candidates)

    cmap = plt.get_cmap("tab20")
    roi_color = {}  # a given ROI keeps the same colour in every panel
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharex=True)
    for ax, (cond, acq, tr) in zip(axes, CONDITIONS):
        if cond not in data:
            ax.set_axis_off()
            continue
        x = data[cond]
        idx = shared if shared is not None else top_rois(x, n, rank, candidates)
        t = (np.arange(x.shape[0]) + 0.5) * tr
        for r in idx:
            if r not in roi_color:
                roi_color[r] = cmap(len(roi_color) % 20)
            ax.plot(t, x[:, r], color=roi_color[r], lw=1.6, marker="o" if x.shape[0] <= 40 else None,
                    ms=3, label=labels[r])
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("z-scored PSC" if zscore else "PSC (%)")
        ax.set_title(f"{cond} (TR={tr:g} s)")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
    how = f"ROIs chosen in {ref}" if select == "reference" else "ROIs chosen per condition"
    scope = "visual ROIs" if len(candidates) < len(labels) else "all ROIs"
    fig.suptitle(f"sub-{sub:02d} - top {n} {scope} by {rank} PSC ({how})", y=1.03)
    fig.tight_layout()
    out.mkdir(parents=True, exist_ok=True)

    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=300)
    (out / f"sub-{sub:02d}_top{n}_timecourses.{fmt}").write_bytes(buf.getvalue())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deriv", required=True, help="derivatives folder containing sub-XX/reg/")
    ap.add_argument("--out", default="figures_top5")
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 20)))
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--rank", choices=["peak", "mean"], default="peak")
    ap.add_argument("--select", choices=["reference", "per-condition"], default="per-condition")
    ap.add_argument("--ref", choices=[c[0] for c in CONDITIONS], default="160bins")
    ap.add_argument("--labels", required=True, help="Schaefer label file (ROI names)")
    ap.add_argument("--rois", choices=["visual", "all"], default="visual",
                    help="candidate ROIs for the top-N selection (default: visual network only)")
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    a = ap.parse_args()

    labels = load_labels(a.labels)
    candidates = np.arange(len(labels))
    if a.rois == "visual":
        candidates = np.array([i for i, l in enumerate(labels) if VIS_REGEX.search(l)])
        if candidates.size == 0:
            raise SystemExit("No 'Vis' ROIs found: check that --labels is the Schaefer label file.")
        print(f"Restricting to {candidates.size} visual ROIs")

    for sub in a.subjects:
        print(f"=== sub-{sub:02d} ===")
        plot_subject(a.deriv, sub, Path(a.out), a.fmt, a.n, a.rank, a.select.replace("-", "-"),
                     a.ref, labels, candidates, a.zscore)
    print(f"Done. Figures in {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
