import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

T1w_image_paths = sorted(Path("/home/jesse/BRICIA/MVH_JPhitidis_PhD/projects/PVS_SCORING/data_new/images").glob("*T1w.nii.gz"))
temp_input_dir = TemporaryDirectory()

for im in T1w_image_paths:
    shutil.copy(im, temp_input_dir.name)

# Run SynthSeg
subprocess.run(
    [
        "python", "/home/jesse/projects/SynthSeg/scripts/commands/SynthSeg_predict.py",
        "--i", temp_input_dir.name,
        "--o", "/home/jesse/BRICIA/MVH_JPhitidis_PhD/projects/PVS_SCORING/data_new/labels",
        "--qc", str(Path(__file__).parent / "qc.csv"),
    ]
)