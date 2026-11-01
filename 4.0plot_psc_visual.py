#!/usr/bin/env python
# Laura Belli, October 2026
# Script to plot PSC in the Visual Network (Yeo's 7 networks) and top 5 ROIs with highest activation
"""
Visual-network PSC: the binnings compared, and the 5 strongest visual ROIs.

For every subject this writes, into --outdir:

  sub-XX_visual_psc_overview.png        Figure 1. Top: mean PSC of the visual network over time, one line per
                                        binning (16, 40, 80 bins), all on the same 0-40 s axis. Bottom: one glass
                                        brain per ROI, for the 5 visual ROIs with the highest PSC.
  sub-XX_visual_top5_timecourses.png    Figure 2. PSC time course of those same 5 ROIs, one panel per binning.
  sub-XX_visual_top5.csv                the ranking behind both figures (table view)
  sub-XX_visual_timecourses.csv         every plotted value, long format (table view)

Time axis: the paradigm lasts --duration s (default 40) whatever the binning, so the time per volume is
duration / n_volumes: 16 -> 2.5 s, 40 -> 1 s, 80 -> 0.5 s.

How the "highest PSC" ROIs are chosen
  * Visual network = parcels whose network name starts with --network (default "Vis"), both hemispheres.
  * Only ROIs present in ALL binnings are used, so the three lines and the five ROIs are comparable.
  * Score of an ROI in one binning = the peak of its PSC time course (--score abs: the largest |PSC|).
  * By default the rank is the mean of those scores over the binnings (--rank-by all); use --rank-by 40 to
    rank on one binning only.

Usage
    python plot_psc_visual.py --subs 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 --bins 16 40 80 160 \
        --labels ../derivatives/templates/Schaefer2018_200Parcels_7Networks_order.txt \
        --atlas  ../derivatives/templates/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz \
        --deriv ../derivatives --outdir psc_figures

  PSC files read:  <deriv>/sub-XX/reg/sub-XX_task-CS_acq-<N>bins_Schaefer_psc.txt
  Column -> parcel matching: if a PSC file has fewer columns than the label table has parcels, the parcels that
  survived at BOLD resolution are read from <deriv>/sub-XX/reg/sub-XX_task-CS_acq-<N>bins_Schaefer2bold.nii.gz.

Requires: numpy, matplotlib; for the glass brains also nibabel and nilearn (pip install nibabel nilearn).
"""
import argparse
import csv
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt 
import numpy as np  
from matplotlib.colors import ListedColormap 
from matplotlib.lines import Line2D

# ---- design tokens (light surface). Series colours are the validated categorical palette, fixed order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]     # slots 1-5
BIN_SLOT = {"16": 0, "40": 1, "80": 2, "160": 3}                    # a binning keeps its colour in every figure
ROI_FILL = INK         


# ----------------------------------------------------------------------------- labels
def read_label_table(path):
    """Schaefer label table -> (ids, names). Accepts '1  7Networks_LH_Vis_1  120 18 134 0' lines and the
    two-line variant ('7Networks_LH_Vis_1' followed by an RGB line)."""
    ids, names = [], []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) >= 2 and p[0].isdigit() and not p[1].isdigit():
            ids.append(int(p[0]))
            names.append(p[1])
        elif len(p) == 1 and "Networks_" in p[0]:
            ids.append(len(ids) + 1)
            names.append(p[0])
    if not ids:
        raise SystemExit(f"no parcel labels found in {path}")
    return np.array(ids), names


def network_of(name):
    """'7Networks_LH_Vis_1' -> 'Vis'."""
    p = name.split("_")
    return p[2] if len(p) > 2 else "NA"


def short_name(name):
    """'7Networks_LH_Vis_3' -> 'LH Vis 3'."""
    p = name.split("_")
    return " ".join(p[1:]) if len(p) > 2 else name


# ----------------------------------------------------------------------------- data
def column_ids(n_cols, all_ids, bold_atlas):
    """Label id of every PSC column. fslmeants --label writes one column per label present, ascending."""
    if n_cols == len(all_ids):
        return np.asarray(all_ids)
    if Path(bold_atlas).exists():
        import nibabel as nib
        b = np.rint(np.asarray(nib.load(str(bold_atlas)).dataobj)).astype(int)
        ids = np.unique(b[b > 0])
        ids = ids[np.isin(ids, all_ids)]
        if len(ids) == n_cols:
            return ids
        raise ValueError(f"{n_cols} columns but {len(ids)} labels in {bold_atlas}")
    raise ValueError(f"{n_cols} columns but {len(all_ids)} labels, and {bold_atlas} was not found "
                     "to tell which parcels were lost")


def load_subject(sub, bins, deriv, all_ids):
    data = {}
    for b in bins:
        reg = Path(deriv) / f"sub-{sub}" / "reg"
        f = reg / f"sub-{sub}_task-CS_acq-{b}bins_200Schaefer_psc.txt"
        if not f.exists() or f.stat().st_size == 0:
            raise ValueError(f"missing {f}")
        psc = np.loadtxt(f)
        psc = psc[:, None] if psc.ndim == 1 else psc
        ids = column_ids(psc.shape[1], all_ids, reg / f"sub-{sub}_task-CS_acq-{b}bins_Schaefer2bold.nii.gz")
        data[b] = {"psc": psc, "ids": ids}
    return data


def select_top(data, bins, name_of, prefix, n_top, score, rank_by):
    """Visual ROIs common to all binnings, ranked by peak PSC. Returns a dict."""
    common = None
    for b in bins:
        s = {int(i) for i in data[b]["ids"] if network_of(name_of[int(i)]).startswith(prefix)}
        common = s if common is None else common & s
    common = sorted(common)
    if not common:
        raise ValueError(f"no '{prefix}' parcels are present in all binnings")
    cols, peaks = {}, {}
    for b in bins:
        idx = {int(i): k for k, i in enumerate(data[b]["ids"])}
        cols[b] = [idx[i] for i in common]
        x = data[b]["psc"][:, cols[b]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            peaks[b] = np.nanmax(np.abs(x) if score == "abs" else x, axis=0)
    rank_bins = bins if rank_by == "all" else [rank_by]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_score = np.nanmean(np.vstack([peaks[b] for b in rank_bins]), axis=0)
    order = [int(i) for i in np.argsort(-np.nan_to_num(mean_score, nan=-np.inf)) if np.isfinite(mean_score[i])]
    order = order[:n_top]
    return {"common": common, "cols": cols, "peaks": peaks, "score": mean_score, "order": order,
            "top_ids": [common[i] for i in order], "rank_bins": rank_bins}


# ----------------------------------------------------------------------------- plotting helpers
def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_facecolor(SURFACE)
    ax.axhline(0, color=INK2, lw=0.8, alpha=0.6, zorder=1)


def time_axis(ax, duration):
    ax.set_xlim(0, duration)
    ax.set_xticks(np.arange(0, duration + 1e-9, 5))


def spread(ys, min_gap):
    """Nudge label positions apart so direct labels never overlap."""
    out = np.array(ys, dtype=float)
    order = np.argsort(out)
    for k in range(1, len(order)):
        i, j = order[k - 1], order[k]
        if out[j] - out[i] < min_gap:
            out[j] = out[i] + min_gap
    return out


def make_brain_drawer(atlas_path):
    """Returns draw(ax, roi_id): the ROI as a filled glass-brain projection (sagittal + axial) in ax."""
    import nibabel as nib
    from nilearn import plotting
    img = nib.load(str(atlas_path))
    data = np.rint(np.asarray(img.dataobj)).astype(int)
    cmap = ListedColormap([ROI_FILL])

    def draw(ax, roi_id):
        vol = (data == roi_id).astype(np.float32)
        if not vol.any():
            raise ValueError(f"label {roi_id} not found in {atlas_path}")
        plotting.plot_glass_brain(
            nib.Nifti1Image(vol, img.affine), axes=ax, display_mode="xz", cmap=cmap, vmin=0, vmax=1,
            threshold=0.5, colorbar=False, plot_abs=False, annotate=False, alpha=0.3,
            resampling_interpolation="nearest")
    return draw


def peak_word(args, bins):
    """'mean peak' when the rank averages the peaks of several binnings, else 'peak'."""
    return "mean peak" if args.rank_by == "all" and len(bins) > 1 else "peak"


# ----------------------------------------------------------------------------- figure 1
def fig_overview(sub, bins, t, vis_mean, vis_sem, n_common, sel, name_of, args, draw_brain):
    with_brain = draw_brain is not None
    fig = plt.figure(figsize=(13, 9.4 if with_brain else 5.6), facecolor=SURFACE)
    left, right = 0.07, 0.90
    top_, bot_ = (0.88, 0.47) if with_brain else (0.80, 0.17)
    ax = fig.add_subplot(fig.add_gridspec(1, 1, left=left, right=right, top=top_, bottom=bot_)[0, 0])
    style_axes(ax)
    duration = args.duration

    ends = []
    for b in bins:
        c = SERIES[BIN_SLOT[b]]
        m, e = vis_mean[b], vis_sem[b]
        ax.fill_between(t[b], m - e, m + e, color=c, alpha=0.15, lw=0, zorder=2)
        ax.plot(t[b], m, color=c, lw=1.5, zorder=3, solid_capstyle="round")
        k = int(np.where(np.isfinite(m))[0][-1])
        ends.append((b, t[b][k], m[k]))
    time_axis(ax, duration)
    ax.set_xlabel("Time (s)", color=INK2, fontsize=10)
    ax.set_ylabel(f"PSC (%), mean of {n_common} visual ROIs", color=INK2, fontsize=10)

    # direct labels at the right edge (text in ink, a colour square carries identity), de-collided
    lo, hi = ax.get_ylim()
    ys = spread([y for _, _, y in ends], 0.075 * (hi - lo))
    for (b, x_end, y_end), y_lab in zip(ends, ys):
        c = SERIES[BIN_SLOT[b]]
        ax.plot([x_end, duration * 1.012], [y_end, y_lab], color=c, lw=0.8, clip_on=False, zorder=2)
        ax.plot([duration * 1.012], [y_lab], marker="s", ms=5, color=c, ls="none", clip_on=False)
        ax.text(duration * 1.03, y_lab, f"{b} bins", va="center", ha="left", color=INK, fontsize=9.5,
                clip_on=False)

    ax.set_title(f"sub-{sub}  -  visual network PSC", loc="left", color=INK, fontsize=13, pad=30)
    handles = [Line2D([0], [0], color=SERIES[BIN_SLOT[b]], lw=1.5,
                      label=f"{b} bins ({duration / len(t[b]):g} s per volume)") for b in bins]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(bins), frameon=False,
              fontsize=9.5, labelcolor=INK2, handlelength=1.6, borderaxespad=0.3)
    if with_brain:
        n = len(sel["top_ids"])
        basis = ("mean of the peak PSC over the binnings" if args.rank_by == "all"
                 else f"peak PSC at {args.rank_by} bins")
        fig.text(left, 0.355, f"The {n} visual ROIs with the highest PSC ({basis}) - glass brain, "
                 "sagittal and axial projections", color=INK2, fontsize=10, ha="left", va="bottom")
        gs = fig.add_gridspec(1, n, left=0.04, right=0.98, top=0.25, bottom=0.02, wspace=0.08)
        for k, (rid, o) in enumerate(zip(sel["top_ids"], sel["order"])):
            axb = fig.add_subplot(gs[0, k])
            axb.axis("off")
            try:
                draw_brain(axb, rid)
            except ValueError as err:
                print(f"  WARNING: {err}")
                axb.text(0.5, 0.5, "ROI not in atlas", ha="center", va="center", color=INK2, fontsize=9,
                         transform=axb.transAxes)
            bb = axb.get_position()
            fig.text(bb.x0 + bb.width / 2, bb.y1 + 0.006,
                     f"#{k + 1}  {short_name(name_of[rid])}\n{peak_word(args, bins)} {sel['score'][o]:.2f} %",
                     ha="center", va="bottom", color=INK, fontsize=10, linespacing=1.4)
    return fig


# ----------------------------------------------------------------------------- figure 2
def fig_top_rois(sub, bins, t, roi_tc, sel, name_of, args):
    n = len(bins)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.5 * n + 1.5), sharex=True, sharey=True, facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80 if n > 1 else 0.70, bottom=0.10 if n > 1 else 0.18,
                        hspace=0.38)
    for ax, b in zip(axes, bins):
        style_axes(ax)
        for k, rid in enumerate(sel["top_ids"]):
            ax.plot(t[b], roi_tc[b][:, k], color=SERIES[k], lw=1.5, solid_capstyle="round", zorder=3)
        time_axis(ax, args.duration)
        ax.set_title(f"{b} bins  ({args.duration / len(t[b]):g} s per volume)", loc="left", color=INK2,
                     fontsize=10, pad=4)
        ax.set_ylabel("PSC (%)", color=INK2, fontsize=10)
    axes[-1].set_xlabel("Time (s)", color=INK2, fontsize=10)
    handles = [Line2D([0], [0], color=SERIES[k], lw=1.5,
                      label=f"#{k + 1}  {short_name(name_of[rid])}  ({peak_word(args, bins)} {sel['score'][o]:.2f} %)")
               for k, (rid, o) in enumerate(zip(sel["top_ids"], sel["order"]))]
    fig.text(0.08, 0.955, f"sub-{sub}  -  PSC of the {len(handles)} visual ROIs with the highest PSC",
             color=INK, fontsize=13, ha="left", va="top")
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.065, 0.925), ncol=3, frameon=False,
               fontsize=9.5, labelcolor=INK2, handlelength=1.6, columnspacing=1.6)
    return fig


# ----------------------------------------------------------------------------- tables
def write_tables(outdir, sub, bins, t, vis_mean, roi_tc, sel, name_of):
    with open(Path(outdir) / f"sub-{sub}_visual_top5.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "label_id", "name"] + [f"peak_psc_{b}bins" for b in bins] + ["score_used_for_rank"])
        for k, (rid, o) in enumerate(zip(sel["top_ids"], sel["order"])):
            w.writerow([k + 1, rid, name_of[rid]] + [f"{sel['peaks'][b][o]:.4f}" for b in bins] +
                       [f"{sel['score'][o]:.4f}"])
    with open(Path(outdir) / f"sub-{sub}_visual_timecourses.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bins", "time_s", "series", "label_id", "name", "psc_percent"])
        for b in bins:
            for i, ti in enumerate(t[b]):
                w.writerow([b, f"{ti:.3f}", "visual_network_mean", "", "", f"{vis_mean[b][i]:.5f}"])
            for k, rid in enumerate(sel["top_ids"]):
                for i, ti in enumerate(t[b]):
                    w.writerow([b, f"{ti:.3f}", f"top{k + 1}", rid, name_of[rid], f"{roi_tc[b][i, k]:.5f}"])


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subs", nargs="+", default=["01", "02", "03", "04", "05"], help="subject numbers")
    ap.add_argument("--bins", nargs="+", default=["16", "40", "80"], help="binnings to compare (max 4)")
    ap.add_argument("--deriv", default="../derivatives")
    ap.add_argument("--labels", default="../derivatives/templates/Schaefer2018_200Parcels_7Networks_order.txt")
    ap.add_argument("--atlas", help="Schaefer atlas in MNI space (for the glass brains)")
    ap.add_argument("--network", default="Vis", help="network name prefix (default Vis)")
    ap.add_argument("--n-top", type=int, default=5)
    ap.add_argument("--score", choices=["peak", "abs"], default="peak")
    ap.add_argument("--rank-by", default="all", help="'all' or one binning, e.g. 40")
    ap.add_argument("--duration", type=float, default=40.0, help="paradigm length in s (default 40)")
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args(argv)

    bins = list(args.bins)
    bad = [b for b in bins if b not in BIN_SLOT]
    if bad:
        ap.error(f"unsupported binning(s) {bad}; supported: {list(BIN_SLOT)}")
    if args.rank_by != "all" and args.rank_by not in bins:
        ap.error("--rank-by must be 'all' or one of --bins")
    if args.n_top > len(SERIES):
        ap.error(f"--n-top is limited to {len(SERIES)} (colour-safe palette)")

    all_ids, names = read_label_table(args.labels)
    name_of = dict(zip(all_ids.tolist(), names))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    draw_brain = None
    if args.atlas:
        try:
            draw_brain = make_brain_drawer(args.atlas)
        except ImportError as err:
            print(f"WARNING: {err}; glass brains skipped (pip install nibabel nilearn)")
    else:
        print("WARNING: no --atlas given; glass brains skipped")

    done, failed = 0, 0
    for sub in args.subs:
        try:
            data = load_subject(sub, bins, args.deriv, all_ids)
            sel = select_top(data, bins, name_of, args.network, args.n_top, args.score, args.rank_by)
        except (ValueError, OSError) as err:
            print(f"sub-{sub}: skipped ({err})")
            failed += 1
            continue
        if len(sel["top_ids"]) < args.n_top:
            print(f"sub-{sub}: WARNING only {len(sel['top_ids'])} '{args.network}' ROIs available")
        if not sel["top_ids"]:
            print(f"sub-{sub}: skipped (no ROI with a finite PSC)")
            failed += 1
            continue

        t, vis_mean, vis_sem, roi_tc = {}, {}, {}, {}
        for b in bins:
            n = data[b]["psc"].shape[0]
            t[b] = np.arange(n) * args.duration / n
            x = data[b]["psc"][:, sel["cols"][b]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vis_mean[b] = np.nanmean(x, axis=1)
                nv = np.maximum(np.sum(np.isfinite(x), axis=1), 1)
                vis_sem[b] = np.nanstd(x, axis=1) / np.sqrt(nv)
            idx = {int(i): k for k, i in enumerate(data[b]["ids"])}
            roi_tc[b] = data[b]["psc"][:, [idx[i] for i in sel["top_ids"]]]

        print(f"sub-{sub}: {len(sel['common'])} {args.network} ROIs in all binnings; top {len(sel['top_ids'])}: " +
              ", ".join(f"{short_name(name_of[r])} ({sel['score'][o]:.2f}%)"
                        for r, o in zip(sel["top_ids"], sel["order"])))

        f1 = fig_overview(sub, bins, t, vis_mean, vis_sem, len(sel["common"]), sel, name_of, args, draw_brain)
        f1.savefig(Path(args.outdir) / f"sub-{sub}_visual_psc_overview.png", dpi=args.dpi, facecolor=SURFACE)
        plt.close(f1)
        f2 = fig_top_rois(sub, bins, t, roi_tc, sel, name_of, args)
        f2.savefig(Path(args.outdir) / f"sub-{sub}_visual_top5_timecourses.png", dpi=args.dpi, facecolor=SURFACE)
        plt.close(f2)
        write_tables(args.outdir, sub, bins, t, vis_mean, roi_tc, sel, name_of)
        done += 1
    print(f"finished: {done} subject(s) done, {failed} skipped; files in {args.outdir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
