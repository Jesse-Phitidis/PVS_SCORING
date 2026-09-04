from pvs_scoring.lightning_modules.base import Base
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import torchio as tio
from collections import defaultdict
import pandas as pd
import datetime
import torch


class Multitask(Base): 
    
    def __init__(self, seg_weight_scheduler, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seg_weight_scheduler = seg_weight_scheduler
        
    def on_train_start(self):
        super().on_train_start()
        df = self.trainer.datamodule.df_train
        n_masks = len(df[df["mask"]=="y"])
        n_scores = len(df[pd.notna(df["bg"])])
        self.base_seg_weight = 2 * (1/n_masks) / ((1/n_masks) + (1/n_scores))
        print(f"base_seg_weight = {self.base_seg_weight}")
        
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        images, labels, masks = self.extract_data_from_batch(batch, train=True)
        assert images.shape[0] == 1, "must use batch size 1"
        has_labels = not labels[0][0] == "N/A"
        has_masks = not torch.any(torch.isnan(masks))
        _, pred, seg = self(images)
        seg = [out for out in seg if out is not None]
        if has_labels:
            loss_classification = (2 - self.base_seg_weight) * (self.criterion[0](pred[0], labels[0]).mean() + self.criterion[1](pred[1], labels[1]).mean()) / 2
        if has_masks:
            loss_segmentation = self.base_seg_weight * self.criterion[2](seg, masks)
        seg_weight = self.seg_weight_scheduler(self.current_epoch, self.trainer.max_epochs)
        if has_labels and has_masks:
            loss = (1 - seg_weight) * loss_classification + seg_weight * loss_segmentation
        elif has_labels and (not has_masks):
            loss = (1 - seg_weight) * loss_classification
        elif (not has_labels) and has_masks:
            loss = seg_weight * loss_segmentation
        else:
            raise Exception
        self.log("loss", loss.detach().item(), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: dict) -> None:
        images, labels, masks = self.extract_data_from_batch(batch)
        feats, pred, seg = self(images)
        feats = [f.detach().cpu() for f in feats]
        pred = [p.detach().cpu().to(dtype=torch.float32) for p in pred]
        seg = [s.detach().cpu() if s is not None else s for s in seg]
        for i in range(2):
            pred[i] = torch.softmax(pred[i], dim=1)
        self.metric[0](pred, [l.cpu() for l in labels])
        seg_last = seg[-1].to(torch.float32)
        seg_last = torch.softmax(seg_last, dim=1)
        seg_last = self.pred_to_output_space(seg_last, batch, as_tensor=True)
        seg_last = torch.argmax(seg_last, dim=1, keepdim=True)
        self.metric[1](seg_last, masks.cpu())
        
    def on_validation_epoch_end(self) -> None:
        metric_dict_classification = self.metric[0].aggregate()
        metric_dict_segmentation = self.metric[1].aggregate()
        self.metric[0].reset(), self.metric[1].reset()
        metric_dict_classification.update(metric_dict_segmentation)
        self.log_dict(metric_dict_classification, on_step=False, on_epoch=True)
        torch.cuda.empty_cache()
    
    def on_predict_start(self):
        super().on_predict_start()
        self.preds = defaultdict(list)
        
    def predict_step(self, batch: dict):
        images = self.extract_data_from_batch(batch, pred=True)
        _, pred, seg = self(images)
        for i in range(2):
            pred[i] = torch.softmax(pred[i], dim=1)
        key = self.trainer.datamodule.sequences[0]
        ID = Path(batch[key]["path"][0]).name.split(f"_{key}")[0]
        self.preds["id"].append(ID)
        for i in range(3):
            self.preds[f"bg_derivative_{i+1}"].append(pred[0][0,i].item())
        for i in range(3):
            self.preds[f"cso_derivative_{i+1}"].append(pred[1][0,i].item())  
        seg_last = seg[-1]
        seg_last = torch.softmax(seg_last, dim=1)
        seg_nib = self.pred_to_output_space(seg_last.detach().cpu(), batch, as_tensor=False)
        key = self.trainer.datamodule.sequences[0]
        nib.save(seg_nib, self.dir_pred / f"{str(Path(batch[key]['path'][0]).name)}".replace(f"_{key}.nii.gz", "_pred.nii.gz"))
            
    def on_predict_end(self):
        super().on_predict_end()
        pd.DataFrame(self.preds).to_csv(self.dir_pred / "predictions.csv", index=False)
        
    def extract_data_from_batch(self, batch: dict, train: bool=False, pred: bool=False) -> tuple[torch.Tensor]:
        images = []
        for key in self.trainer.datamodule.sequences + self.trainer.datamodule.input_masks:
            images.append(batch[key]["data"].to(torch.float32))
        images = torch.cat(images, dim=1)
        if pred:
            return images
        masks = batch["PVS"]["data"].to(torch.float32)
        if train: # Background channel must be added for multiclass training
            bg = 1 - masks
            masks = torch.cat([bg, masks], dim=1)
        if isinstance(batch["BG_derivative"], torch.Tensor):
            BG_derivative_label = batch["BG_derivative"].to(dtype=torch.long) - 1
            CSO_derivative_label = batch["CSO_derivative"].to(dtype=torch.long) - 1
        else:
            BG_derivative_label = batch["BG_derivative"]
            CSO_derivative_label = batch["CSO_derivative"]
        labels = [BG_derivative_label, CSO_derivative_label]
        return images, labels, masks   
    
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
    
    
def log_batch_memory_usage(batch, batch_idx=None, prefix="BATCH", log_file="batch_memory_log.txt"):
    """
    Logs CUDA memory usage (MB) for each tensor in a nested dictionary batch,
    and writes the output to a file.

    Parameters:
        batch (dict): The batch dictionary.
        batch_idx (int, optional): Index of the batch (for logging).
        prefix (str): Label prefix for logging.
        log_file (str): File path to append the log output.
    """
    def tensor_size(tensor):
        return tensor.element_size() * tensor.nelement() / 1e6  # bytes to MB

    total_cuda_mem = 0.0
    details = []
    log_lines = []

    def process_entry(key, val, parent_key=""):
        nonlocal total_cuda_mem

        if isinstance(val, torch.Tensor) and val.is_cuda:
            mem = tensor_size(val)
            total_cuda_mem += mem
            details.append(f"{parent_key + key}: {mem:.2f} MB | shape: {tuple(val.shape)}")

        elif isinstance(val, dict):
            for subkey, subval in val.items():
                process_entry(subkey, subval, parent_key=key + ".")

    for key, val in batch.items():
        process_entry(key, val)

    tag = f"[{prefix} {batch_idx}]" if batch_idx is not None else f"[{prefix}]"

    subject_id = batch["PVS"]["path"][0].split("/")[-1]
    
    log_lines.append(f"{tag} CUDA memory usage: {total_cuda_mem:.2f} MB")
    log_lines.append(f"ID: {subject_id}")
    for d in details:
        log_lines.append(f"  - {d}")

    # Print to console
    for line in log_lines:
        print(line)

    # Append to log file
    with open(log_file, "a") as f:
        for line in log_lines:
            f.write(line + "\n")
        

def log_gpu_memory(tag="", logfile=None):
    """Logs current GPU memory usage with optional tag and logging to file."""
    if not torch.cuda.is_available():
        return

    device = torch.cuda.current_device()
    reserved = torch.cuda.memory_reserved(device) / 1024**2
    allocated = torch.cuda.memory_allocated(device) / 1024**2
    max_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    max_allocated = torch.cuda.max_memory_allocated(device) / 1024**2

    timestamp = datetime.datetime.now().strftime("%H:%M:%S")
    msg = (
        f"[{timestamp}] {tag} | "
        f"Allocated: {allocated:.2f} MB | "
        f"Reserved: {reserved:.2f} MB | "
        f"Max Allocated: {max_allocated:.2f} MB | "
        f"Max Reserved: {max_reserved:.2f} MB"
    )

    if logfile:
        with open(logfile, "a") as f:
            f.write(msg + "\n")
    else:
        print(msg)