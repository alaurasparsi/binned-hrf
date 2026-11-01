#!/usr/bin/env bash
# Laura Belli, October 2026
# script to preprocess T1w data
export PATH=~/ants-2.6.5-ubuntu-24.04-X64-gcc/ants-2.6.5/bin:$PATH
export PATH=~/c3d-1.1.0-Linux-x86_64/bin:$PATH

for sub in $(seq -f %02g 11 20); do
    #set dir        
    adir="../derivatives/sub-${sub}/anat"
    anat="sub-${sub}_T1w"
    echo "Processing subject ${sub}"

    # 1. bias field correction on T1
    # to install ImageMath: you need ANTs suite, version for ubuntu 24.04 from here: https://github.com/ANTsX/ANTs/releases
    # then unzip and run: export PATH=~/ants-2.6.5-ubuntu-24.04-X64-gcc/ants-2.6.5/bin:$PATH
    echo "Truncating Image Intensity and running N4BiasFieldCorrection on ${anat}"
    ImageMath 3 "${adir}/${anat}_trunc.nii.gz" TruncateImageIntensity "${adir}/${anat}-RAS.nii.gz" 0.02 0.98 256
    N4BiasFieldCorrection -d 3 -i "${adir}/${anat}_trunc.nii.gz" -o "${adir}/${anat}_bfc.nii.gz"

    # 2. Skull stripping
    echo "Skull Stripping ${anat}"
    anat_in="${anat}_bfc"
    3dSkullStrip -input ${adir}/${anat_in}.nii.gz \
    -prefix ${adir}/${anat}_brain.nii.gz \
    -orig_vol

    # 3. Mask
    echo "Masking ${anat}"
    fslmaths ${adir}/${anat}_brain.nii.gz -bin ${adir}/${anat}_brain-mask.nii.gz

    # 4. segmentation
    anat_in="${anat}_brain"
    Atropos -d 3 -a "${adir}/${anat_in}.nii.gz" \
    -o "${adir}/${anat}_seg.nii.gz" \
    -x "${adir}/${anat}_brain-mask.nii.gz" -i kmeans[3] \
    --use-partial-volume-likelihoods \
    -v 1

    # 5. registration T1 -> MNI
    # T1w is 1mm isotropic, reference atlas is MNI152lin_T1_1mm_brain.nii.gz from here ~/fsl/data/linearMNI
    adir="../derivatives/sub-${sub}/anat"
    anat="sub-${sub}_T1w"
    rdir="../derivatives/sub-${sub}/reg"
    sdir="../derivatives/templates"
    std="MNI152lin_T1_1mm"
    fixed="${sdir}/${std}_brain.nii.gz"
    moving="${adir}/${anat}_brain.nii.gz"
    pfx="${rdir}/sub-${sub}_T1w2MNI_"

    echo "Normalizing ${anat} to ${std}"

    antsRegistration -d 3 -v 1 -z 1 \
        -r "[${fixed},${moving},1]" \
        -o "[${pfx},${pfx}Warped.nii.gz,${pfx}InverseWarped.nii.gz]" \
        -x "[${sdir}/${std}_brain_mask.nii.gz,${adir}/${anat}_brain-mask.nii.gz]" \
        -n Linear -u 0 -w "[0.005,0.995]" \
        -t Rigid[0.1] \
            -m "MI[${fixed},${moving},1,48,Regular,0.1]" \
            -c "[1000x500x250x100,1e-6,10]" -f 8x4x2x1 -s 3x2x1x0vox \
        -t Affine[0.1] \
            -m "MI[${fixed},${moving},1,48,Regular,0.1]" \
            -c "[1000x500x250x100,1e-6,10]" -f 8x4x2x1 -s 3x2x1x0vox \
        -t "SyN[0.1,3,0]" \
            -m "CC[${fixed},${moving},1,5]" \
            -c "[100x70x50x20,1e-6,10]" -f 8x4x2x1 -s 3x2x1x0vox
        
done
