from pvs_scoring.lightning_modules.base import Base
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import torchio as tio

class Segmentor(Base): 
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def training_step(self, batch: dict) -> torch.Tensor:
        images, labels = self.extract_data_from_batch(batch, train=True)
        pred = self(images)
        loss = self.criterion(pred, labels)
        self.log("loss", loss.item(), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: dict) -> None:
        images, labels = self.extract_data_from_batch(batch)
        pred = torch.softmax(self(images), dim=1)
        pred = self.pred_to_output_space(pred, batch, as_tensor=True)
        pred = torch.argmax(pred, dim=1, keepdim=True)
        self.metric(pred, labels)
        
    def predict_step(self, batch: dict):
        images = self.extract_data_from_batch(batch, pred=True)
        pred = torch.softmax(self(images), dim=1)
        pred_nib = self.pred_to_output_space(pred, batch, as_tensor=False)
        key = self.trainer.datamodule.sequences[0]
        nib.save(pred_nib, self.dir_pred / f"{str(Path(batch[key]['path'][0]).name)}".replace(f"_{key}.nii.gz", "_pred.nii.gz"))
        
    def extract_data_from_batch(self, batch: dict, train: bool=False, pred: bool=False) -> tuple[torch.Tensor]:
        images = []
        for key in self.trainer.datamodule.sequences + self.trainer.datamodule.input_masks:
            images.append(batch[key]["data"].to(torch.float32))
        images = torch.cat(images, dim=1)
        if pred:
            return images
        labels = batch["PVS"]["data"].to(torch.float32)
        if train: # Background channel must be added for multiclass training
            bg = 1 - labels
            labels = torch.cat([bg, labels], dim=1)
        return images, labels
    
    def pred_to_output_space(self, pred: torch.Tensor, batch: dict, as_tensor: bool) -> None:
        dtype, device = pred.dtype, pred.device
        key = self.trainer.datamodule.sequences[0]
        pred_affine = batch[key]["affine"][0]
        target_affine = batch["PVS"]["affine"][0]
        target_shape = batch["PVS"]["data"].shape[2:]
        pred = tio.LabelMap(tensor=pred.cpu()[0], affine=pred_affine.cpu().numpy())
        resampler = tio.Resample(target=(target_shape, np.array(target_affine.cpu())), label_interpolation="linear")
        pred = resampler(pred)["data"].unsqueeze(0).to(dtype=dtype, device=device)
        if as_tensor: # Return as tensor i.e. during validation
            return pred
        # Otherwise return as nib with correct affine and header (soft prediction)
        ori_lab = nib.load(batch[key]["path"][0])
        pred = nib.Nifti1Image(pred[0,1,...].cpu().numpy(), affine=ori_lab.affine, header=ori_lab.header)
        return pred
        
    