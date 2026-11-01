#!/usr/bin/env python
# Laura Belli, October 2026
# Script to fit and plot canonical, delayed and free HRF models; models comparison

"""
Does the response to the 15 s ON / 25 s OFF block match a canonical (EPI-like) HRF?

Model-based analysis of the folded 40 s cycle (mean PSC over the visual ROIs, per subject
and per temporal resolution):

   y(t) = A * [ boxcar(0-15 s)  (*)  h(t) ] + offset        ((*) = periodic convolution, 40 s)

 - Periodic convolution: blocks repeat every 40 s, so the undershoot of one block spills into
   the start of the next. A free offset replaces the baseline-window choice.
 - Bin averaging: each bin averages the model over its width (TR = 40/bins s).
 - h = double-gamma HRF (SPM form): gamma(shape p1, scale 1) - c * gamma(shape p2, scale 1).
   SPM canonical = p1 6, p2 16, c 1/6 (peak ~5 s, undershoot ~15-16 s).

Three nested models per curve:
   canonical : SPM HRF fixed; free A, offset                           (k = 2)
   delay     : SPM HRF with a free time shift                          (k = 3)
               -> shift = delay of the response relative to canonical
   free      : free p1, p2, c (peak delay, undershoot delay, undershoot ratio)  (k = 5)

Derived HRF descriptors (from the fitted single-event HRF h):
   TTP, FWHM, undershoot time, undershoot/peak ratio.  Reference = the SPM canonical HRF.

Outputs (in --out):
   fit_results.csv          every fit (subject, condition, model, parameters, R2, adjR2, AICc...)
   group_summary.csv        group median + bootstrap 95% CI (over subjects) vs canonical reference
   fig_A_group_fit.<fmt>    group-mean curve +- SEM with the mean fitted models
   fig_B_subject_fits_<cond>.<fmt>   per-subject fits (sanity check)
   fig_C_hrf_parameters.<fmt>        fitted HRF descriptors per condition, canonical dashed
   fig_D_model_comparison.<fmt>      adjusted R2 and dAICc (free - canonical)
   stats_model_comparison.csv        the three pairwise comparisons (delay vs canonical, free vs
                                     delay, free vs canonical): paired Wilcoxon per condition on
                                     adjR2 and AICc, Holm over all tests, effect size, and Friedman
                                     test of the comparison x condition interaction (also printed)

Usage:
  python 6.0fit_hrf_model.py --deriv ../derivatives --out ../derivatives/hrf_fit \
    --labels ../derivatives/templates/Schaefer2018_200Parcels_7Networks_order.txt \
    --fmt svg --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
    --conds 16 40 80 160bins --stim-on 15

Requires: scipy (pip install scipy)
"""
import argparse
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import friedmanchisquare, gamma, rankdata, wilcoxon

warnings.simplefilter("ignore", RuntimeWarning)

CONDITIONS = [  # (label, n bins, TR in seconds)
    ("16bins", 16, 2.5),
    ("40bins", 40, 1.0),
    ("80bins", 80, 0.5),
    ("160bins", 160, 0.25),
]
COND = {c: (a, tr) for c, a, tr in CONDITIONS}
PSC_PATTERN = ("{deriv}/sub-{sub:02d}/reg/"
               "sub-{sub:02d}_task-CS_acq-{acq}bins_200Schaefer_psc{suffix}.txt")
COLORS = {"16bins": "#d95f02", "40bins": "#7570b3", "80bins": "#1b9e77", "160bins": "#1f78b4"}
MODELS = ["canonical", "delay", "free"]
MODEL_COLORS = {"canonical": "0.45", "delay": "#1b9e77", "free": "#d62728"}
VIS_REGEX = re.compile(r"(^|_)Vis")

CYCLE = 40.0
DT = 0.05                                  # fine grid; 40/0.05 = 800 divisible by every bin count
N_FINE = int(round(CYCLE / DT))
T_FINE = np.arange(N_FINE) * DT
T_LONG = np.arange(3 * N_FINE) * DT        # 120 s, folded onto the 40 s cycle
SPM = dict(p1=6.0, p2=16.0, c=1.0 / 6.0, shift=0.0)
BOX = None                                 # set by set_stim()

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


# ----------------------------------------------------------------- model
def set_stim(stim_on):
    global BOX
    BOX = (T_FINE < stim_on).astype(float)


def hrf_fn(t, p1, p2, c, shift=0.0):
    x = np.asarray(t, dtype=float) - shift
    h = gamma.pdf(x, p1, scale=1.0) - c * gamma.pdf(x, p2, scale=1.0)
    h[x <= 0] = 0.0
    return h


def cycle_model(p1, p2, c, shift):
    """Unit-amplitude periodic response to the boxcar, on the fine 40 s grid."""
    h = hrf_fn(T_LONG, p1, p2, c, shift).reshape(3, N_FINE).sum(axis=0)   # fold onto the cycle
    return np.fft.irfft(np.fft.rfft(BOX) * np.fft.rfft(h), n=N_FINE) * DT


def bin_avg(m, acq):
    return m.reshape(acq, -1).mean(axis=1)


def ols(x, y):
    X = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, y - X @ coef


def hrf_metrics(p1, p2, c, shift):
    t = np.arange(0, 60, 0.01)
    h = hrf_fn(t, p1, p2, c, shift)
    ipk = int(np.argmax(h))
    pk = h[ipk]
    out = dict(TTP=t[ipk], FWHM=np.nan, UnderT=np.nan, UnderRatio=np.nan)
    if pk <= 0:
        return out
    half = pk / 2.0
    left = np.flatnonzero(h[:ipk + 1] < half)
    right = np.flatnonzero(h[ipk:] < half)
    if right.size:
        t_l = t[left[-1] + 1] if left.size else t[0]
        out["FWHM"] = t[ipk + right[0]] - t_l
    iu = ipk + int(np.argmin(h[ipk:]))
    if h[iu] < 0 and iu < len(t) - 1:
        out["UnderT"] = t[iu]
        out["UnderRatio"] = -h[iu] / pk
    else:
        out["UnderRatio"] = 0.0
    return out


REF = hrf_metrics(**SPM)


def fit_curve(y, acq, model):
    ok = np.isfinite(y)
    n = int(ok.sum())
    if n < 8:
        return None

    def build(params):
        if model == "canonical":
            p = (SPM["p1"], SPM["p2"], SPM["c"], 0.0)
        elif model == "delay":
            p = (SPM["p1"], SPM["p2"], SPM["c"], float(params[0]))
        else:
            p = (float(params[0]), float(params[1]), float(params[2]), 0.0)
        return p, cycle_model(*p)

    def resid(params):
        _, m = build(params)
        return ols(bin_avg(m, acq)[ok], y[ok])[1]

    if model == "canonical":
        best_p, k = np.array([]), 2
    elif model == "delay":
        grid = np.arange(-4, 4.01, 0.25)
        s0 = grid[int(np.argmin([np.sum(resid([s]) ** 2) for s in grid]))]
        best_p = least_squares(resid, [s0], bounds=([-5.0], [5.0])).x
        k = 3
    else:
        best = None
        for p1 in (4.0, 6.0, 8.0):
            for p2 in (12.0, 16.0, 22.0):
                for c in (0.1, 0.3):
                    sol = least_squares(resid, [p1, p2, c],
                                        bounds=([2.0, 8.0, 0.0], [12.0, 30.0, 0.6]))
                    if best is None or sol.cost < best.cost:
                        best = sol
        best_p, k = best.x, 5

    p, m = build(best_p)
    coef, res = ols(bin_avg(m, acq)[ok], y[ok])
    rss = float(np.sum(res ** 2))
    tss = float(np.sum((y[ok] - y[ok].mean()) ** 2))
    r2 = 1 - rss / tss if tss > 0 else np.nan
    adj = 1 - (1 - r2) * (n - 1) / (n - k) if n > k else np.nan
    aic = n * np.log(rss / n) + 2 * k
    aicc = aic + 2 * k * (k + 1) / (n - k - 1) if n - k - 1 > 0 else np.nan
    row = dict(model=model, A=coef[0], offset=coef[1], p1=p[0], p2=p[1], c=p[2], shift=p[3],
               n=n, k=k, RSS=rss, R2=r2, adjR2=adj, RMSE=np.sqrt(rss / n), AICc=aicc,
               **hrf_metrics(*p))
    return row, coef[0] * m + coef[1]


# ----------------------------------------------------------------- data
def load_vis_mask(path, n_roi=200):
    if not path or not Path(path).exists():
        return None
    labs = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            labs.append(parts[1] if len(parts) > 1 and parts[0].isdigit() else parts[0])
    if len(labs) < n_roi:
        print(f"[warn] label file has {len(labs)} entries (< {n_roi}); ignoring it")
        return None
    vis = np.array([bool(VIS_REGEX.search(l)) for l in labs[:n_roi]])
    print(f"Labels loaded: {vis.sum()} visual ROIs out of {n_roi}")
    return vis


def load_curve(deriv, sub, acq, suffix, vis):
    path = Path(PSC_PATTERN.format(deriv=deriv, sub=sub, acq=acq, suffix=suffix))
    if not path.exists():
        print(f"[skip] missing {path}")
        return None
    x = np.genfromtxt(path)
    y = np.nanmean(x[:, vis], axis=1) if vis is not None else np.nanmean(x, axis=1)
    if len(y) != acq:
        print(f"[warn] sub-{sub:02d} {acq} bins: file has {len(y)} rows; skipped")
        return None
    return y


def save_fig(fig, path, fmt):
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()  # via memory: avoids "w+b" open failing on OneDrive/WSL
    fig.savefig(buf, format=fmt, bbox_inches="tight", dpi=300)
    path.write_bytes(buf.getvalue())
    plt.close(fig)


def boot_ci(v, rng, n_boot):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 4:
        return np.nan, np.nan, np.nan
    idx = rng.integers(0, len(v), (n_boot, len(v)))
    med = np.median(v[idx], axis=1)
    lo, hi = np.percentile(med, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)


# ----------------------------------------------------------------- plots
def plot_group_fit(datas, preds, conds, vis_label, out, fmt):
    fig, axes = plt.subplots(1, len(conds), figsize=(3.6 * len(conds), 3.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, cond in zip(axes, conds):
        acq, tr = COND[cond]
        Y = np.array(datas[cond])
        t = (np.arange(acq) + 0.5) * tr
        mu, se = np.nanmean(Y, axis=0), np.nanstd(Y, axis=0, ddof=1) / np.sqrt(len(Y))
        ax.axvspan(0, STIM_ON, color="0.93", zorder=0)
        ax.axhline(0, color="k", lw=0.5, ls=":")
        ax.fill_between(t, mu - se, mu + se, color=COLORS[cond], alpha=0.3, lw=0)
        ax.plot(t, mu, "o", color=COLORS[cond], ms=2.5 if acq > 40 else 4, label="data (mean ± SEM)")
        for m in MODELS:
            if preds.get((cond, m)):
                ax.plot(T_FINE, np.mean(preds[(cond, m)], axis=0), color=MODEL_COLORS[m],
                        lw=1.6, ls="--" if m == "canonical" else "-", label=m)
        ax.set_title(f"{cond} (n={len(Y)})")
        ax.set_xlabel("Time from stimulus onset (s)")
    axes[0].set_ylabel("PSC (%)")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Group fit of the block response ({vis_label}); shaded: stimulus ON", y=1.03)
    fig.tight_layout()
    save_fig(fig, out / f"fig_A_group_fit.{fmt}", fmt)


def plot_subject_fits(cond, subj_data, out, fmt):
    acq, tr = COND[cond]
    t = (np.arange(acq) + 0.5) * tr
    n = len(subj_data)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.8 * ncol, 2.2 * nrow), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for ax, (sub, y, fits, adj) in zip(axes, subj_data):
        ax.axvspan(0, STIM_ON, color="0.93", zorder=0)
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.plot(t, y, "o", color=COLORS[cond], ms=2.5)
        for m in ("canonical", "free"):
            if m in fits:
                ax.plot(T_FINE, fits[m], color=MODEL_COLORS[m], lw=1.2,
                        ls="--" if m == "canonical" else "-")
        ax.set_title(f"sub-{sub:02d}  adjR²={adj:.2f}" if np.isfinite(adj) else f"sub-{sub:02d}",
                     fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes[n:]:
        ax.set_axis_off()
    fig.suptitle(f"{cond}: per-subject fits (dashed grey: canonical, red: free HRF)", y=1.0)
    fig.tight_layout()
    save_fig(fig, out / f"fig_B_subject_fits_{cond}.{fmt}", fmt)


def plot_parameters(df, conds, out, fmt):
    panels = [("free", "TTP", "HRF time to peak (s)", REF["TTP"]),
              ("free", "FWHM", "HRF FWHM (s)", REF["FWHM"]),
              ("free", "UnderT", "Undershoot time (s)", REF["UnderT"]),
              ("free", "UnderRatio", "Undershoot / peak", REF["UnderRatio"]),
              ("delay", "shift", "Delay vs canonical (s)", 0.0)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6))
    rng = np.random.default_rng(0)
    for ax, (model, col, lab, ref) in zip(axes, panels):
        d = df[df["model"] == model]
        piv = d.pivot(index="subject", columns="condition", values=col).reindex(columns=conds)
        for _, r in piv.iterrows():
            ax.plot(range(len(conds)), r.values, color="0.85", lw=0.6, zorder=1)
        for i, c in enumerate(conds):
            v = piv[c].dropna().values
            ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=14, color=COLORS[c],
                       alpha=0.85, zorder=3)
            if len(v):
                ax.hlines(np.median(v), i - 0.25, i + 0.25, color="k", lw=2, zorder=4)
        ax.axhline(ref, color="crimson", ls="--", lw=1.2, zorder=2)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(conds, fontsize=8)
        ax.set_ylabel(lab)
    fig.suptitle("Fitted HRF descriptors (dot = subject, black = median, red dashed = SPM canonical)")
    fig.tight_layout()
    save_fig(fig, out / f"fig_C_hrf_parameters.{fmt}", fmt)


def plot_model_comparison(df, conds, out, fmt):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    ax = axes[0]
    for mi, m in enumerate(MODELS):
        vals = [df[(df["model"] == m) & (df["condition"] == c)]["adjR2"].dropna().values
                for c in conds]
        pos = np.arange(len(conds)) + (mi - 1) * 0.26
        bp = ax.boxplot(vals, positions=pos, widths=0.22, showfliers=False, patch_artist=True,
                        medianprops=dict(color="k"))
        for b in bp["boxes"]:
            b.set_facecolor(MODEL_COLORS[m])
            b.set_alpha(0.65)
        ax.plot([], [], color=MODEL_COLORS[m], lw=6, label=m)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_ylabel("Adjusted R²")
    ax.legend(fontsize=7, frameon=False)

    ax = axes[1]
    rng = np.random.default_rng(0)
    can = df[df["model"] == "canonical"].set_index(["subject", "condition"])["AICc"]
    fre = df[df["model"] == "free"].set_index(["subject", "condition"])["AICc"]
    d_aicc = (fre - can).dropna()
    for i, c in enumerate(conds):
        v = d_aicc[d_aicc.index.get_level_values("condition") == c].values
        ax.scatter(i + rng.uniform(-0.08, 0.08, len(v)), v, s=14, color=COLORS[c], alpha=0.85)
        if len(v):
            ax.hlines(np.median(v), i - 0.25, i + 0.25, color="k", lw=2)
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(-2, color="crimson", ls="--", lw=1)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds)
    ax.set_ylabel("ΔAICc = free - canonical\n(< -2: free HRF clearly better)")
    fig.suptitle("Does the canonical HRF explain the response, or is a free shape needed?")
    fig.tight_layout()
    save_fig(fig, out / f"fig_D_model_comparison.{fmt}", fmt)


# ----------------------------------------------------------------- statistics
def holm(p):
    """Holm-Bonferroni adjusted p-values (NaNs ignored)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return out


def rank_biserial(d):
    """Matched-pairs rank-biserial correlation (+1: all differences positive)."""
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return np.nan
    r = rankdata(np.abs(d))
    wp, wm = r[d > 0].sum(), r[d < 0].sum()
    return (wp - wm) / (wp + wm)


def _wilcoxon_p(d):
    d = np.asarray(d, dtype=float)
    return wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan


PAIRS = [("canonical", "delay"), ("delay", "free"), ("canonical", "free")]   # (simpler, richer)


def model_comparison_stats(df, conds):
    """The three pairwise model comparisons (richer minus simpler):
         delay vs canonical : does a latency shift alone improve the fit?
         free  vs delay     : does the HRF shape matter beyond the shift?
         free  vs canonical : overall
    For each pair and condition: paired Wilcoxon on adjusted R2 and on AICc, with Holm correction
    across ALL pair x condition tests (separately for adjR2 and AICc). Also a Friedman test, per pair,
    of the paired difference across conditions (comparison x condition interaction).
    adjR2 diff > 0 = richer model fits better; dAICc < 0 = richer model fits better."""
    adj = df.pivot_table(index=["subject", "condition"], columns="model", values="adjR2")
    aic = df.pivot_table(index=["subject", "condition"], columns="model", values="AICc")
    rows = []
    for simple, rich in PAIRS:
        if simple not in adj.columns or rich not in adj.columns:
            continue
        for cond in conds:
            a = adj[adj.index.get_level_values("condition") == cond][[simple, rich]].dropna()
            b = aic[aic.index.get_level_values("condition") == cond][[simple, rich]].dropna()
            d_adj = (a[rich] - a[simple]).values
            d_aic = (b[rich] - b[simple]).values
            rows.append(dict(
                comparison=f"{rich} vs {simple}", condition=cond, n=len(d_adj),
                adjR2_simple=a[simple].median() if len(a) else np.nan,
                adjR2_rich=a[rich].median() if len(a) else np.nan,
                median_diff_adjR2=np.median(d_adj) if len(d_adj) else np.nan,
                rank_biserial_adjR2=rank_biserial(d_adj), p_adjR2=_wilcoxon_p(d_adj),
                median_dAICc=np.median(d_aic) if len(d_aic) else np.nan,
                p_dAICc=_wilcoxon_p(d_aic)))
    res = pd.DataFrame(rows)
    if res.empty:
        return res, {}
    res["p_adjR2_holm"] = holm(res["p_adjR2"].values)
    res["p_dAICc_holm"] = holm(res["p_dAICc"].values)

    inter = {}
    for simple, rich in PAIRS:
        if simple not in adj.columns or rich not in adj.columns:
            continue
        for name, piv in (("adjR2", adj), ("AICc", aic)):
            wide = (piv[rich] - piv[simple]).unstack("condition").reindex(columns=conds).dropna()
            if len(wide) >= 4 and len(conds) >= 3:
                stat, p = friedmanchisquare(*[wide[c].values for c in conds])
                inter[(f"{rich} vs {simple}", name)] = (len(wide), stat, p)
    return res, inter


# ----------------------------------------------------------------- main
def main():
    global STIM_ON
    ap = argparse.ArgumentParser()
    ap.add_argument("--deriv", required=True, help="derivatives folder containing sub-XX/reg/")
    ap.add_argument("--out", default="hrf_fit")
    ap.add_argument("--labels", default=None, help="Schaefer label file (visual ROI mean)")
    ap.add_argument("--subjects", type=int, nargs="*", default=list(range(1, 20)))
    ap.add_argument("--conds", nargs="*", default=[c[0] for c in CONDITIONS],
                    choices=[c[0] for c in CONDITIONS])
    ap.add_argument("--psc-suffix", default="")
    ap.add_argument("--stim-on", type=float, default=15.0, help="stimulus ON duration (s)")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    a = ap.parse_args()

    STIM_ON = a.stim_on
    set_stim(STIM_ON)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = load_vis_mask(a.labels)
    vis_label = "visual ROI mean" if vis is not None else "all-ROI mean"
    conds = [c for c in [x[0] for x in CONDITIONS] if c in a.conds]

    print(f"Canonical reference: TTP {REF['TTP']:.2f} s, FWHM {REF['FWHM']:.2f} s, "
          f"undershoot at {REF['UnderT']:.2f} s, undershoot/peak {REF['UnderRatio']:.2f}")

    rows, datas, preds = [], {c: [] for c in conds}, {}
    subj_fits = {c: [] for c in conds}
    for sub in a.subjects:
        print(f"=== sub-{sub:02d} ===")
        for cond in conds:
            acq, tr = COND[cond]
            y = load_curve(a.deriv, sub, acq, a.psc_suffix, vis)
            if y is None:
                continue
            datas[cond].append(y)
            fits, adj_free = {}, np.nan
            for model in MODELS:
                r = fit_curve(y, acq, model)
                if r is None:
                    continue
                row, fine = r
                rows.append(dict(subject=f"sub-{sub:02d}", condition=cond, TR=tr, **row))
                preds.setdefault((cond, model), []).append(fine)
                fits[model] = fine
                if model == "free":
                    adj_free = row["adjR2"]
            subj_fits[cond].append((sub, y, fits, adj_free))

    if not rows:
        raise SystemExit("No PSC files found: check --deriv and --psc-suffix.")
    df = pd.DataFrame(rows)
    (out / "fit_results.csv").write_text(df.to_csv(index=False))

    plot_group_fit(datas, preds, conds, vis_label, out, a.fmt)
    for cond in conds:
        if subj_fits[cond]:
            plot_subject_fits(cond, subj_fits[cond], out, a.fmt)
    plot_parameters(df, conds, out, a.fmt)
    plot_model_comparison(df, conds, out, a.fmt)

    # ---- group summary vs canonical reference
    rng = np.random.default_rng(1)
    srows = []
    spec = [("free", "TTP", REF["TTP"]), ("free", "FWHM", REF["FWHM"]),
            ("free", "UnderT", REF["UnderT"]), ("free", "UnderRatio", REF["UnderRatio"]),
            ("delay", "shift", 0.0)]
    for cond in conds:
        for model, col, ref in spec:
            v = df[(df["model"] == model) & (df["condition"] == cond)][col].values
            med, lo, hi = boot_ci(v, rng, a.n_boot)
            srows.append(dict(condition=cond, model=model, parameter=col, n=int(np.isfinite(v).sum()),
                              median=med, ci_lo=lo, ci_hi=hi, canonical_reference=ref,
                              reference_inside_CI=bool(lo <= ref <= hi) if np.isfinite(lo) else None))
    summ = pd.DataFrame(srows)
    (out / "group_summary.csv").write_text(summ.to_csv(index=False))

    pd.set_option("display.width", 160)
    print("\nGroup medians with bootstrap 95% CI over subjects vs SPM canonical:")
    print(summ.round(2).to_string(index=False))

    print("\nGoodness of fit (median adjusted R2):")
    print(df.groupby(["condition", "model"])["adjR2"].median().unstack().reindex(conds).round(3))
    can = df[df["model"] == "canonical"].set_index(["subject", "condition"])["AICc"]
    fre = df[df["model"] == "free"].set_index(["subject", "condition"])["AICc"]
    d = (fre - can).dropna()
    print("\nSubjects where the free HRF is clearly better than canonical (dAICc < -2):")
    for cond in conds:
        v = d[d.index.get_level_values("condition") == cond]
        print(f"  {cond:<8} {int((v < -2).sum())}/{len(v)}")
    stats, inter = model_comparison_stats(df, conds)
    (out / "stats_model_comparison.csv").write_text(stats.to_csv(index=False))
    print("\nPairwise model comparisons (richer vs simpler) - paired Wilcoxon per condition,")
    print("Holm-corrected across all comparison x condition tests (separately for adjR2 and AICc).")
    print("adjR2 diff > 0 = richer model fits better; dAICc < 0 = richer model fits better;")
    print("rank_biserial: +1/-1 = all subjects change in the same direction, 0 = no effect:")
    print(stats.round(4).to_string(index=False))
    for (pair, name), (n, stat, p) in inter.items():
        print(f"Comparison x condition interaction ({pair}, Friedman on {name}, n={n}): "
              f"chi2={stat:.2f}, p={p:.4g}")
    if not inter:
        print("Interaction test skipped (needs >= 3 conditions and >= 4 complete subjects).")
    print(f"\nDone. Outputs in {out.resolve()}")


if __name__ == "__main__":
    main()
