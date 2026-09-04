from pathlib import Path
import nibabel as nib
import torchio as tio
import numpy as np
from tempfile import TemporaryDirectory
import subprocess
import pandas as pd
from tempfile import TemporaryDirectory
from tqdm import tqdm
import shutil

def save_nii(nii: nib.Nifti1Image, path: Path, dtype: str, is_label: bool):
    assert "uint" in dtype, "dtype shoudl be uint8 or uint16"
    bits = int(dtype.split("uint")[-1])
    max_value = (2 ** bits) - 1
    if not is_label:
        nii = tio.RescaleIntensity(out_min_max=(0, max_value))(nii)
    data = nii.get_fdata()
    assert np.min(data) >= 0 and np.max(data) <= max_value
    data = data.astype(dtype)
    nii = nib.Nifti1Image(data, nii.affine, nii.header)
    nii.set_data_dtype(dtype)
    nib.save(nii, path)

def nib_load(path: Path | str, lazy: bool = True) -> nib.Nifti1Image:
    if lazy:
        return nib.load(path)
    else:
        nii = nib.load(path)
        return nib.Nifti1Image(nii.get_fdata(), nii.affine, nii.header)

def brain_extraction(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    """ This function assumes freesurfer is set up correctly"""
    tempdir = TemporaryDirectory()
    nii_in_path = Path(tempdir.name) / "in.nii.gz"
    nii_out_path = Path(tempdir.name) / "out.nii.gz"
    nii_mask_path = Path(tempdir.name) / "mask.nii.gz"
    nib.save(nii, nii_in_path)
    subprocess.run(["mri_synthstrip", "-i", str(nii_in_path), "-o", str(nii_out_path), "-m", str(nii_mask_path)], stdout=subprocess.DEVNULL)
    return nib_load(nii_mask_path, lazy=False)


def __main__():
    
    T2w_image_paths = sorted(Path("/home/jesse/BRICIA/MVH_JPhitidis_PhD/projects/PVS_SCORING/data_new/images").glob("*T2w.nii.gz"))
    for T2w_path in tqdm(T2w_image_paths):

        save_path = str(T2w_path).replace("images", "labels").replace("T2w.nii.gz", "BRAIN.nii.gz")

        nii = nib_load(T2w_path)
        mask = brain_extraction(nii)
        
        save_nii(mask, Path(save_path), "uint8", is_label=True)
        
if __name__ == "__main__":
    __main__()
    