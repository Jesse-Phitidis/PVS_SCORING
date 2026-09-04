from pathlib import Path
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_dilation, zoom
from nipype.interfaces import niftyreg
from tempfile import TemporaryDirectory

########## functions ##########

def nib_load(path: Path | str, lazy: bool = True) -> nib.Nifti1Image:
    if lazy:
        return nib.load(path)
    else:
        nii = nib.load(path)
        return nib.Nifti1Image(nii.get_fdata(), nii.affine, nii.header)


def NiftyRegNlin(
    ref: nib.Nifti1Image, 
    moving: nib.Nifti1Image, 
    labels: list[nib.Nifti1Image] | None, 
    interp_labels: str = "NN"    
    ):
    
    temp_dir = TemporaryDirectory()
    
    nib.save(ref, Path(temp_dir.name) / "ref.nii.gz")
    nib.save(moving, Path(temp_dir.name) / "moving.nii.gz")
    
    # Do initial affine registration
    reg_aff = niftyreg.RegAladin()
    reg_aff.inputs.ref_file = Path(temp_dir.name) / "ref.nii.gz"
    reg_aff.inputs.flo_file = Path(temp_dir.name) / "moving.nii.gz"
    reg_aff.inputs.res_file = Path(temp_dir.name) / "trash.nii.gz"
    reg_aff.inputs.aff_file = Path(temp_dir.name) / "affine.txt"
    reg_aff.inputs.verbosity_off_flag = True
    reg_aff.run()
    
    # Do non linear registration
    reg_nlin = niftyreg.RegF3D()
    reg_nlin.inputs.ref_file = Path(temp_dir.name) / "ref.nii.gz"
    reg_nlin.inputs.flo_file = Path(temp_dir.name) / "moving.nii.gz"
    reg_nlin.inputs.res_file = Path(temp_dir.name) / "moving.nii.gz"
    reg_nlin.inputs.aff_file = Path(temp_dir.name) / "affine.txt"
    reg_nlin.inputs.cpp_file = Path(temp_dir.name) / "nlin_transform.nii.gz"
    reg_nlin.inputs.verbosity_off_flag = True
    reg_nlin.run()
    
    moving_out = nib_load(Path(temp_dir.name) / "moving.nii.gz", lazy=False)
    
    # Apply transformation to labels
    labels_out = []
    for i, label in enumerate(labels):
        nib.save(label, Path(temp_dir.name) / "label.nii.gz")
        reg_res = niftyreg. RegResample()
        reg_res.inputs.ref_file = Path(temp_dir.name) / "ref.nii.gz"
        reg_res.inputs.flo_file = Path(temp_dir.name) / "label.nii.gz"
        reg_res.inputs.out_file = Path(temp_dir.name) / "label.nii.gz"
        reg_res.inputs.trans_file = Path(temp_dir.name) / "nlin_transform.nii.gz"
        reg_res.inputs.inter_val = interp_labels
        reg_res.inputs.verbosity_off_flag = True
        reg_res.run()
        label_out = nib_load(Path(temp_dir.name) / "label.nii.gz", lazy=False)
        labels_out.append(label_out)
        
    return moving_out, labels_out


########## main ##########


T1w_paths = sorted(Path("/home/jesse/BRICIA/MVH_JPhitidis_PhD/projects/PVS_SCORING/data_new/images").glob("*T1w.nii.gz"))

bg_labels = [49, 10, 50, 11, 51, 12, 52, 13, 26, 58]
wm_labels = [2, 41]

for T1w_path in tqdm(T1w_paths, total=len(T1w_paths)):

    BG_out_path = str(T1w_path).replace("/images/", "/labels/").replace("_T1w", "_BG")
    CSO_out_path = str(T1w_path).replace("/images/", "/labels/").replace("_T1w", "_CSO")
    
    if Path(BG_out_path).exists() and Path(CSO_out_path).exists():
        print(f"Already exists: {BG_out_path} and {CSO_out_path}, skipping.")
        continue

    T1w = nib.load(T1w_path)

    # Register template
    template = NiftyRegNlin(
        ref=T1w,
        moving=nib.load(Path(__file__).parent / "template_73y.nii.gz"),
        labels=[nib.load(Path(__file__).parent / "template_73y_artefactual_PVSROI.nii.gz")],
    )[1][0].get_fdata()

    # Load seg
    seg_full = nib.load(str(T1w_path).replace("/images/", "/labels/").replace("_T1w", "_synthseg")).get_fdata()

    # BG
    bg = np.zeros_like(seg_full)
    for l in bg_labels:
        bg[seg_full==l]=1
    f = np.array(T1w.header.get_zooms()) / np.array([1,1,1])
    bg = zoom(bg, f, order=0)
    bg = binary_dilation(bg, iterations=6)
    bg = zoom(bg, 1/f, order=0)
    assert bg.shape == template.shape
    bg[template==1]=0
    for l in np.unique(seg_full):
        if l not in bg_labels + wm_labels:
            bg[seg_full==l]=0

    # CSO
    cso=np.zeros_like(bg)
    for l in wm_labels:
        cso[seg_full==l]=1
    cso[bg==1]=0
    cso[template==1]=0

    # Save
    bg_mask = nib.Nifti1Image(bg.astype(np.uint8), header=T1w.header, affine=T1w.affine)
    cso_mask = nib.Nifti1Image(cso.astype(np.uint8), header=T1w.header, affine=T1w.affine)
    nib.save(bg_mask, BG_out_path)
    nib.save(cso_mask, CSO_out_path)




