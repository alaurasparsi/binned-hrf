#!/usr/bin/env bash
# Laura Belli, October 2026
# script to coregister anatomical to functional space and extract BOLD timeseries according to 200 ROIs Schaefer
# functional data have a spatial resolution of 1mm isotropic, and variable temporal resolution (0.25-2.5 s)
#
#   per subject:         Schaefer atlas MNI -> T1w      ANTs, inverse of your T1w->MNI transforms, MultiLabel)
#                        grey-matter mask in T1w      
#   per binning:         BOLD mean + bet -> BOLD -> T1w FLIRT, rigid and its inverse
#                        atlas T1w -> BOLD              FLIRT, nearest neighbour
#                        GM fraction T1w -> BOLD        FLIRT, trilinear -> threshold -> mask the atlas
#                        mean time series per parcel    fslmeants on the original 4D data, no resampling needed
#
# Outputs per subject/binning in <deriv>/sub-XX/reg:
#   sub-XX_task-CS_acq-Nbins_Schaefer2bold.nii.gz        atlas GM-masked in BOLD space
#   sub-XX_task-CS_acq-Nbins_Schaefer_timeseries.txt     volumes x parcels = one column per label present
#   sub-XX_task-CS_acq-Nbins_bold2T1w.nii.gz             QC: BOLD mean in T1w space

set -euo pipefail

# ------------------------------ configuration ------------------------------
subs="01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19"
binnings="16 40 80 160"
deriv="../derivatives"
atlas="../derivatives/templates/Schaefer2018_200Parcels_7Networks_order_FSLMNI152_1mm.nii.gz"
n_parcels=200                  # expected number of parcels
xfm_tag="2MNI_"                # T1w->MNI transforms: <rdir>/sub-XX_T1w${xfm_tag}{0GenericAffine.mat,1InverseWarp.nii.gz}
use_gm=1                       # 1 = keep only grey-matter voxels in each parcel
gm_label=2                     # grey-matter value in your seg (Atropos: 1=CSF 2=GM 3=WM)
gm_thr=0.5                     # keep BOLD voxels with at least this GM fraction
threads=2                      # ANTs threads
# -----------------------------------------------------------------------------------

force=0
while [ $# -gt 0 ]; do
    case "$1" in
        --subs) subs="$2"; shift 2;;
        --bins) binnings="$2"; shift 2;;
        --force) force=1; shift;;
        *) echo "unknown option: $1"; exit 2;;
    esac
done
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${threads}" FSLOUTPUTTYPE=NIFTI_GZ
dry="${DRY_RUN:-0}"

log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { echo "+ $*"; [ "${dry}" = 1 ] && return 0; nice -n 10 "$@"; }
valid() { [ -s "$1" ] || return 1; case "$1" in *.gz) gzip -t "$1" 2>/dev/null;; esac; }
# todo <output> [inputs...]: true if the output is missing/corrupt or older than any input
todo() {
    local out="$1" i; shift
    [ "${force}" = 1 ] && return 0
    valid "${out}" || return 0
    for i in "$@"; do [ "${i}" -nt "${out}" ] && return 0; done
    return 1
}
tmpof() { case "$1" in *.nii.gz) echo "${1%.nii.gz}.tmp.nii.gz";; *) echo "${1%.*}.tmp.${1##*.}";; esac; }
# atomic <final> <cmd...>: @OUT@ marks where the output goes; renamed only if the command succeeds
atomic() {
    local final="$1"; shift
    local tmp a; local -a args=(); tmp="$(tmpof "${final}")"
    for a in "$@"; do args+=("${a//@OUT@/${tmp}}"); done
    run "${args[@]}" || { rm -f "${tmp}"; return 1; }
    [ "${dry}" = 1 ] || mv -f "${tmp}" "${final}"
}

[ "${dry}" = 1 ] || [ -e "${atlas}" ] || { log "atlas not found: ${atlas}"; exit 1; }
ok=0; bad=0

for sub in ${subs}; do
    adir="${deriv}/sub-${sub}/anat"
    fdir="${deriv}/sub-${sub}/func"
    rdir="${deriv}/sub-${sub}/reg"
    anat="sub-${sub}_T1w"
    t1_brain="${adir}/${anat}_brain.nii.gz"
    seg="${adir}/${anat}_seg.nii.gz"
    t1_to_std_aff="${rdir}/${anat}${xfm_tag}0GenericAffine.mat"
    std_to_t1_warp="${rdir}/${anat}${xfm_tag}1InverseWarp.nii.gz"

    if [ "${dry}" = 0 ]; then
        miss=0; need=("${t1_brain}" "${t1_to_std_aff}" "${std_to_t1_warp}")
        [ "${use_gm}" = 1 ] && need+=("${seg}")
        for f in "${need[@]}"; do [ -e "${f}" ] || { log "sub-${sub}: missing ${f}"; miss=1; }; done
        if [ "${miss}" = 1 ]; then log "sub-${sub}: skipped"; bad=$((bad+1)); continue; fi
    fi

    # ---- once per subject: atlas in T1w space and GM mask in T1w space ----
    atlas_t1="${rdir}/sub-${sub}_Schaefer_in_T1w.nii.gz"
    gm_t1="${rdir}/sub-${sub}_GM_in_T1w.nii.gz"
    if todo "${atlas_t1}" "${atlas}" "${t1_brain}" "${t1_to_std_aff}" "${std_to_t1_warp}"; then
        log "sub-${sub}: atlas MNI -> T1w"
        atomic "${atlas_t1}" antsApplyTransforms -d 3 -i "${atlas}" -r "${t1_brain}" -o @OUT@ \
            -n MultiLabel -t "[${t1_to_std_aff},1]" -t "${std_to_t1_warp}"
    fi
    if [ "${use_gm}" = 1 ] && todo "${gm_t1}" "${seg}"; then
        atomic "${gm_t1}" fslmaths "${seg}" -thr "${gm_label}" -uthr "${gm_label}" -bin @OUT@
    fi

    # ---- per binning ----
    for acq in ${binnings}; do
        log "sub-${sub}, ${acq} bins"
        func="sub-${sub}_task-CS_acq-${acq}bins_bold"
        bold_4d="${fdir}/${func}-RAS.nii.gz"
        bold_avg="${fdir}/${func}_avg.nii.gz"
        bold_brain="${fdir}/${func}_avg_brain.nii.gz"
        pfx="${rdir}/sub-${sub}_task-CS_acq-${acq}bins"
        mat="${pfx}_bold2T1w.mat"; mat_inv="${pfx}_T1w2bold.mat"
        bold_in_t1="${pfx}_bold2T1w.nii.gz"
        atlas_raw="${pfx}_Schaefer2bold_raw.nii.gz"
        gm_frac="${pfx}_GMfrac2bold.nii.gz"; gm_b="${pfx}_GM2bold.nii.gz"
        atlas_b="${pfx}_Schaefer2bold.nii.gz"
        ts="${pfx}_Schaefer_timeseries.txt"

        if [ "${dry}" = 0 ] && [ ! -e "${bold_4d}" ]; then log "  missing ${bold_4d}; skipped"; bad=$((bad+1)); continue; fi

        # 1. mean volume + brain extraction
        todo "${bold_avg}" "${bold_4d}" && atomic "${bold_avg}" fslmaths "${bold_4d}" -Tmean @OUT@
        todo "${bold_brain}" "${bold_avg}" && atomic "${bold_brain}" bet "${bold_avg}" @OUT@

        # 2. BOLD -> T1w, rigid and inverse
        if todo "${mat}" "${bold_brain}" "${t1_brain}" || todo "${bold_in_t1}" "${mat}"; then
            log "  flirt BOLD -> T1w"
            run flirt -in "${bold_brain}" -ref "${t1_brain}" -dof 6 \
                -omat "${mat}.tmp" -out "$(tmpof "${bold_in_t1}")"
            if [ "${dry}" = 0 ]; then mv -f "${mat}.tmp" "${mat}"; mv -f "$(tmpof "${bold_in_t1}")" "${bold_in_t1}"; fi
        fi
        if todo "${mat_inv}" "${mat}"; then
            run convert_xfm -inverse -omat "${mat_inv}.tmp" "${mat}"
            [ "${dry}" = 1 ] || mv -f "${mat_inv}.tmp" "${mat_inv}"
        fi

        # 3. atlas T1w -> BOLD (labels: nearest neighbour, never interpolate)
        todo "${atlas_raw}" "${atlas_t1}" "${mat_inv}" "${bold_brain}" && \
            atomic "${atlas_raw}" flirt -in "${atlas_t1}" -ref "${bold_brain}" -applyxfm -init "${mat_inv}" \
                -interp nearestneighbour -out @OUT@

        # 4. GM mask in BOLD space: resample the binary GM (trilinear -> GM fraction per BOLD voxel), threshold,
        #    then keep only those voxels in the atlas. Using the fraction keeps partial-volume voxels instead
        #    of only the voxels whose centre falls on grey matter, which matters for small/limbic parcels.
        if [ "${use_gm}" = 1 ]; then
            todo "${gm_frac}" "${gm_t1}" "${mat_inv}" "${bold_brain}" && \
                atomic "${gm_frac}" flirt -in "${gm_t1}" -ref "${bold_brain}" -applyxfm -init "${mat_inv}" \
                    -interp trilinear -out @OUT@
            todo "${gm_b}" "${gm_frac}" && atomic "${gm_b}" fslmaths "${gm_frac}" -thr "${gm_thr}" -bin @OUT@
            todo "${atlas_b}" "${atlas_raw}" "${gm_b}" && atomic "${atlas_b}" fslmaths "${atlas_raw}" -mas "${gm_b}" @OUT@
        else
            todo "${atlas_b}" "${atlas_raw}" && atomic "${atlas_b}" fslmaths "${atlas_raw}" @OUT@
        fi

        # 5. mean time series per parcel on the original 4D data
        todo "${ts}" "${atlas_b}" "${bold_4d}" && \
            atomic "${ts}" fslmeants -i "${bold_4d}" --label="${atlas_b}" -o @OUT@

        if [ "${dry}" = 0 ]; then
            ncols=$(awk 'NR==1{print NF}' "${ts}")
            msg="  ${ncols} parcels in the time series"
            [ "${ncols}" -lt "${n_parcels}" ] && msg="${msg}  (WARNING: ${n_parcels} expected; $((n_parcels-ncols)) parcels have no voxels at this resolution)"
            log "${msg}"
        fi
        ok=$((ok+1))
    done
done
log "finished: ${ok} binnings done, ${bad} skipped or failed"
[ "${bad}" = 0 ]
