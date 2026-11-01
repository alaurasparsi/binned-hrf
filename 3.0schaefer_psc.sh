#!/usr/bin/env bash
# Laura Belli, October 2026
# Script to compute percent signal change (PSC) for each 200 ROIs Schaefer atlas
#   PSC(t) = 100 * (S(t) - baseline) / baseline         with baseline = last 10 seconds of trial.
#
# The trial lasts 40 s whatever the binning, so TR = 40 / bins and the baseline is 10 / TR volumes:
#   16 bins (TR 2.5 s) -> last 4 volumes     40 bins (TR 1 s)    -> last 10 volumes
#   80 bins (TR 0.5 s) -> last 20 volumes   160 bins (TR 0.25 s) -> last 40 volumes
#
# Usage: bash 3.0schaefer_psc.sh [derivatives_dir]      (default: ../derivatives)

set -euo pipefail


binnings=("16" "40" "80" "160")
deriv="${1:-../derivatives}"
trial_s=40      # trial duration (s): 15 s ON + 25 s OFF
base_s=10       # baseline duration (s)

# Last 10 seconds of the trial as the baseline volumes
for sub in $(seq -f %02g 1 19); do
    rdir="${deriv}/sub-${sub}/reg"

    for acq in "${binnings[@]}"; do
        # number of baseline volumes = base_s / TR = acq * base_s / trial_s
        if (( (acq * base_s) % trial_s != 0 )); then
            echo "[warn] ${acq} bins: ${base_s} s is not a whole number of volumes, rounding down" >&2
        fi
        n_base=$(( acq * base_s / trial_s ))
        (( n_base < 1 )) && n_base=1
        base_end=${acq}
        base_start=$(( acq - n_base + 1 ))
        pfx="${rdir}/sub-${sub}_task-CS_acq-${acq}bins"
        ts="${pfx}_Schaefer_timeseries.txt"
        psc="${pfx}_200Schaefer_psc_new.txt"

        echo "PSC: sub-${sub}, ${acq} bins, baseline = last ${n_base} volumes (${base_s} s)"

        awk -v b0="${base_start}" -v b1="${base_end}" '
            { for (j = 1; j <= NF; j++) x[NR, j] = $j; nf = NF }
            END {
                if (b1 == 0 || b1 > NR) b1 = NR
                for (j = 1; j <= nf; j++) {
                    s = 0; n = 0
                    for (i = b0; i <= b1; i++) { s += x[i, j]; n++ }
                    m[j] = s / n
                }
                for (i = 1; i <= NR; i++) {
                    line = ""
                    for (j = 1; j <= nf; j++) {
                        v = (m[j] == 0) ? "nan" : sprintf("%.6f", 100 * (x[i, j] - m[j]) / m[j])
                        line = line (j > 1 ? " " : "") v
                    }
                    print line
                }
            }' "${ts}" > "${psc}"
    done
done
