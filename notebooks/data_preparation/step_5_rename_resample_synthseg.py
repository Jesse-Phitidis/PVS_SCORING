from pathlib import Path
from tqdm import tqdm
import torchio as tio
import nibabel as nib

seg_paths = sorted(Path("/home/jesse/BRICIA/MVH_JPhitidis_PhD/projects/PVS_SCORING/data_new/labels").glob("*synthseg.nii.gz"))

for seg_path in tqdm(seg_paths, total=len(seg_paths)):
    ref = Path(str(seg_path).replace("/labels/", "/images/").replace("_synthseg", ""))
    T = tio.Resample(target=ref, image_interpolation="nearest")
    seg = T(nib.load(seg_path))
    nib.save(seg, str(seg_path).replace("_T1w", ""))
    seg_path.unlink()