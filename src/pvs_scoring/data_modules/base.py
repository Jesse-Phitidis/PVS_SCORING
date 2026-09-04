import pytorch_lightning as pl
import torchio as tio
import torch
import nibabel as nib
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

class Base(pl.LightningDataModule):
    
    def __init__(
        self,
        data_dir: str,
        train_csv: str,
        val_csv: str,
        pred_csv: str,
        transforms_train: tio.Transform | None,
        transforms_predict: tio.Transform | None,
        sequences: list[str],
        input_masks: list[str],
        mask_only_frac: float = 0.0,
        score_only_frac: float = 0.0,
        num_workers: int = 8,
        batch_size: int = 1,
    ):
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.df_train = self.get_train_df(pd.read_csv(train_csv), mask_only_frac, score_only_frac)
        self.df_val = pd.read_csv(val_csv)
        self.df_pred = pd.read_csv(pred_csv)
        self.transforms_train = transforms_train
        self.transforms_predict = transforms_predict
        self.sequences = sequences
        self.input_masks = input_masks
        self.num_workers = num_workers
        self.batch_size = batch_size

    def get_train_df(self, df: pd.DataFrame, mask_only_frac: float, score_only_frac: float) -> pd.DataFrame:
        df["dataset"] = df["id"].apply(self.get_dataset)
        df_both = df[(df["mask"]=="y")&(pd.notna(df["bg"]))]
        df_mask_only = df[(df["mask"]=="y")&(pd.isna(df["bg"]))]
        df_score_only = df[(df["mask"]=="n")&(pd.notna(df["bg"]))]
        df_score_only["total_derivative"] = df_score_only["bg_derivative"] + df_score_only["cso_derivative"]
        df_mask_only_sampled = df_mask_only.groupby("dataset").apply(lambda x: x.sample(frac=mask_only_frac, random_state=1)).reset_index(drop=True)
        df_score_only_sampled = df_score_only.groupby("total_derivative").apply(lambda x: x.sample(frac=score_only_frac, random_state=1)).reset_index(drop=True)
        df_score_only_sampled = df_score_only_sampled.drop(columns=["total_derivative"])
        df = pd.concat([df_both, df_mask_only_sampled, df_score_only_sampled], ignore_index=True)
        df = df.drop(columns=["dataset"])
        return df
    
    def get_subjects(self, df: pd.DataFrame) -> tio.SubjectsDataset:
        subjects = []
        for row in df.iterrows():
            subject_dict = {}
            ID = row[1]["id"]
            for seq in self.sequences:
                subject_dict[seq] = tio.ScalarImage(self.data_dir / "images" / f"{ID}_{seq}.nii.gz")
            for input_mask in self.input_masks + ["BRAIN"]:
                subject_dict[input_mask] = tio.LabelMap(self.data_dir / "labels" / f"{ID}_{input_mask}.nii.gz")
            if row[1]["mask"] == "y":
                subject_dict["PVS"] = tio.LabelMap(self.data_dir / "labels" / f"{ID}_PVS.nii.gz")
            else:
                ref = nib.load(self.data_dir / "labels" / f"{ID}_BRAIN.nii.gz")
                nans = torch.nan * torch.ones((1, *ref.shape))
                subject_dict["PVS"] = tio.LabelMap(tensor=nans, affine=ref.affine)
            if pd.notna(row[1]["bg_derivative"]):
                subject_dict["BG_derivative"] = row[1]["bg_derivative"]
                subject_dict["CSO_derivative"] = row[1]["cso_derivative"]
            else:
                subject_dict["BG_derivative"] = "N/A"
                subject_dict["CSO_derivative"] = "N/A"

            subject = tio.Subject(subject_dict)
            subjects.append(subject)
        return subjects
        
    def setup(self, stage: str) -> None:
        if stage.lower() == "fit":
            subjects = self.get_subjects(self.df_train)
            self.dataset_train = tio.SubjectsDataset(subjects=subjects, transform=self.transforms_train)
            subjects = self.get_subjects(self.df_val)
            self.dataset_val = tio.SubjectsDataset(subjects=subjects, transform=self.transforms_predict)
        if stage.lower() == "predict":
            subjects = self.get_subjects(self.df_pred)
            self.dataset_pred = tio.SubjectsDataset(subjects=subjects, transform=self.transforms_predict)
            
    def train_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset_val, batch_size=1, shuffle=False, num_workers=self.num_workers)
    
    def predict_dataloader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset_pred, batch_size=1, shuffle=False, num_workers=self.num_workers)
    
    @staticmethod
    def get_dataset(ID):
        if ID.startswith("MSSA"):
            return "MSSA"
        if ID.startswith("MSSB"):
            return "MSSB"
        if ID.startswith("MSS3"):
            return "MSS3"
        if ID.startswith("LBC"):
            return "LBC"
        if ID.startswith("VALDO"):
            return "VALDO"