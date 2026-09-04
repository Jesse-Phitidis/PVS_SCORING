from pathlib import Path
import nibabel as nib
import torchio as tio
import numpy as np


def get_min_max_nonzero_indices(nii: nib.Nifti1Image) -> tuple[np.ndarray, np.ndarray]:
    data = nii.get_fdata()
    data_nonzero = np.where(data > 0)
    min_indices = np.min(data_nonzero, axis=1)
    max_indices = np.max(data_nonzero, axis=1)
    return min_indices, max_indices


def get_crop_transform(min_indices: np.ndarray, max_indices: np.ndarray, shape: tuple) -> tio.Crop:
    crop_sides = []
    for i, (low, high) in enumerate(zip(min_indices, max_indices)):
        for e in (low, shape[i]-1-high):
            crop_sides.append(e)
    return tio.Crop(crop_sides)


def brain_mask_and_tissue_norm(nii: nib.Nifti1Image, mask: nib.Nifti1Image) -> nib.Nifti1Image:
    data = nii.get_fdata()
    mask_data = mask.get_fdata()
    min_val = np.min(data[mask_data > 0])
    max_val = np.max(data[mask_data > 0])
    data = (data - min_val) / (max_val - min_val)
    data = data * mask_data
    nii = nib.Nifti1Image(data, nii.affine, nii.header)
    return nii


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
    

def ensure_3D(nii: nib.Nifti1Image) -> nib.Nifti1Image:
    data = nii.get_fdata()
    if len(data.shape) == 3:
        return nii
    elif len(data.shape) == 4:
        data = np.squeeze(data)
        return nib.Nifti1Image(data, nii.affine, nii.header)
    else:
        raise ValueError("Input NIfTI image must be 3D or 4D.")