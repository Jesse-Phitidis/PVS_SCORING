from pathlib import Path
import nibabel as nib
import torchio as tio
from tqdm import tqdm
from pputils import get_min_max_nonzero_indices, get_crop_transform, brain_mask_and_tissue_norm, save_nii, ensure_3D

images_dir = Path("/home/jesse/BRICIA/MVH_JPhitidis_PhD/projects/PVS_SCORING/data_new/images")
labels_dir = Path("/home/jesse/BRICIA/MVH_JPhitidis_PhD/projects/PVS_SCORING/data_new/labels")

ids = [x.name.split("_T2w")[0] for x in sorted(images_dir.glob("*T2w.nii.gz"))]

for ID in tqdm(ids, total=len(ids)):

    # Load T1w, T2w, FLAIR, BRAIN, BG, CSO, synthseg, PVS (maybe)
    T1w = ensure_3D(nib.load(images_dir / f"{ID}_T1w.nii.gz"))
    T2w = ensure_3D(nib.load(images_dir / f"{ID}_T2w.nii.gz"))
    FLAIR = ensure_3D(nib.load(images_dir / f"{ID}_FLAIR.nii.gz"))
    BRAIN = ensure_3D(nib.load(labels_dir / f"{ID}_BRAIN.nii.gz"))
    BG = ensure_3D(nib.load(labels_dir / f"{ID}_BG.nii.gz"))
    CSO = ensure_3D(nib.load(labels_dir / f"{ID}_CSO.nii.gz"))
    synthseg = ensure_3D(nib.load(labels_dir / f"{ID}_synthseg.nii.gz"))
    try:
        PVS = ensure_3D(nib.load(labels_dir / f"{ID}_PVS.nii.gz"))
    except FileNotFoundError:
        PVS = None

    # Zero bg and renormalize all images within brain
    T1w = brain_mask_and_tissue_norm(T1w, BRAIN)
    T2w = brain_mask_and_tissue_norm(T2w, BRAIN)
    FLAIR = brain_mask_and_tissue_norm(FLAIR, BRAIN)

    # Get crop transform from BRAIN
    min_indices, max_indices = get_min_max_nonzero_indices(BRAIN)
    crop_transform = get_crop_transform(min_indices, max_indices, BRAIN.shape)

    # Apply to all
    T1w = crop_transform(T1w)
    T2w = crop_transform(T2w)
    FLAIR = crop_transform(FLAIR)
    BRAIN = crop_transform(BRAIN)
    BG = crop_transform(BG)
    CSO = crop_transform(CSO)
    synthseg = crop_transform(synthseg)
    if PVS is not None:
        PVS = crop_transform(PVS)

    # Save all in correct format
    save_nii(nii=T1w, path=images_dir / f"{ID}_T1w.nii.gz", dtype="uint16", is_label=False)
    save_nii(nii=T2w, path=images_dir / f"{ID}_T2w.nii.gz", dtype="uint16", is_label=False)
    save_nii(nii=FLAIR, path=images_dir / f"{ID}_FLAIR.nii.gz", dtype="uint16", is_label=False)
    save_nii(nii=BRAIN, path=labels_dir / f"{ID}_BRAIN.nii.gz", dtype="uint8", is_label=True)
    save_nii(nii=BG, path=labels_dir / f"{ID}_BG.nii.gz", dtype="uint8", is_label=True)
    save_nii(nii=CSO, path=labels_dir / f"{ID}_CSO.nii.gz", dtype="uint8", is_label=True)
    save_nii(nii=synthseg, path=labels_dir / f"{ID}_synthseg.nii.gz", dtype="uint8", is_label=True)
    if PVS is not None:
        save_nii(nii=PVS, path=labels_dir / f"{ID}_PVS.nii.gz", dtype="uint8", is_label=True)