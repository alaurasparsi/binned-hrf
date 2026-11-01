#!/usr/bin/env python
# Laura Belli, October 2026
# script to reorient nifti files to RAS+ orientation (standard)
#
import nibabel as nib
from nibabel.orientations import (
    io_orientation,
    axcodes2ornt,
    ornt_transform,
    apply_orientation,
    inv_ornt_aff,
)

def reorient_to_ras(infile, outfile):
    img = nib.load(infile)

    # Orientation implied by the NIfTI affine
    current = io_orientation(img.affine)

    print("Current orientation:",
          nib.orientations.ornt2axcodes(current))

    # Desired orientation
    target = axcodes2ornt(("R", "A", "S"))

    # Transformation current -> RAS
    transform = ornt_transform(current, target)

    # Reorder/flip the voxel array
    data = apply_orientation(img.get_fdata(), transform)

    # Corresponding affine
    affine = img.affine @ inv_ornt_aff(transform, img.shape)

    # Make a new image
    out = nib.Nifti1Image(data, affine, img.header)

    # Explicitly update qform/sform
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)

    nib.save(out, outfile)

    print("New orientation:",
          nib.orientations.aff2axcodes(affine))


for sub in range(17, 21):
    reorient_to_ras(
       f"../derivatives/sub-{sub:02d}/anat/sub-{sub:02d}_T1w.nii",
       f"../derivatives/sub-{sub:02d}/anat/sub-{sub:02d}_T1w-RAS.nii.gz"
    )

    reorient_to_ras(
        f"../derivatives/sub-{sub:02d}/func/sub-{sub:02d}_task-CS_acq-16bins_bold.nii.gz",
        f"../derivatives/sub-{sub:02d}/func/sub-{sub:02d}_task-CS_acq-16bins_bold-RAS.nii.gz"
    )

    reorient_to_ras(
            f"../derivatives/sub-{sub:02d}/func/sub-{sub:02d}_task-CS_acq-40bins_bold.nii.gz",
            f"../derivatives/sub-{sub:02d}/func/sub-{sub:02d}_task-CS_acq-40bins_bold-RAS.nii.gz"
    )

    reorient_to_ras(
            f"../derivatives/sub-{sub:02d}/func/sub-{sub:02d}_task-CS_acq-80bins_bold.nii.gz",
            f"../derivatives/sub-{sub:02d}/func/sub-{sub:02d}_task-CS_acq-80bins_bold-RAS.nii.gz"
    )

    reorient_to_ras(
            f"../derivatives/sub-{sub:02d}/func/sub-{sub:02d}_task-CS_acq-160bins_bold.nii.gz",
            f"../derivatives/sub-{sub:02d}/func/sub-{sub:02d}_task-CS_acq-160bins_bold-RAS.nii.gz"
    )
    

