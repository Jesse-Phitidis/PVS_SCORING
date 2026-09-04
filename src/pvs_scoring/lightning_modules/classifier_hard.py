from pvs_scoring.lightning_modules.base import Base
import torch
from pathlib import Path
from collections import defaultdict
import pandas as pd

class Classifier(Base): 
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
    def training_step(self, batch: dict) -> torch.Tensor:
        images, labels = self.extract_data_from_batch(batch)
        _, pred = self(images)
        loss = (self.criterion[0](pred[0], labels[0]).mean() + self.criterion[1](pred[1], labels[1]).mean()) / 2
        self.log("loss", loss.item(), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch: dict) -> None:
        images, labels = self.extract_data_from_batch(batch)
        feats, pred = self(images)
        feats = [f.detach().cpu() for f in feats]
        pred = [p.detach().cpu().to(dtype=torch.float32) for p in pred]
        for i in range(2):
            pred[i] = torch.softmax(pred[i], dim=1)
        self.metric(pred, [l.cpu() for l in labels])
    
    def on_predict_start(self):
        super().on_predict_start()
        self.preds = defaultdict(list)
        
    def predict_step(self, batch: dict):
        images = self.extract_data_from_batch(batch, pred=True)
        feats, pred = self(images)
        feats = [f.detach().cpu() for f in feats]
        pred = [p.detach().cpu() for p in pred]
        for i in range(2):
            pred[i] = torch.softmax(pred[i], dim=1)
        key = self.trainer.datamodule.sequences[0]
        ID = Path(batch[key]["path"][0]).name.split(f"_{key}")[0]
        self.preds["id"].append(ID)
        for i in range(3):
            self.preds[f"bg_derivative_{i+1}"].append(pred[0][0,i].item())
        for i in range(3):
            self.preds[f"cso_derivative_{i+1}"].append(pred[1][0,i].item())
            
    def on_predict_end(self):
        super().on_predict_end()
        pd.DataFrame(self.preds).to_csv(self.dir_pred / "predictions.csv", index=False)
        
    def extract_data_from_batch(self, batch: dict, pred: bool=False) -> tuple[torch.Tensor]:
        images = []
        for key in self.trainer.datamodule.sequences + self.trainer.datamodule.input_masks:
            if key == "pred":
                images.append((batch[key]["data"]>0.5).to(torch.float32))
            else:
                images.append(batch[key]["data"].to(torch.float32))
        images = torch.cat(images, dim=1)
        if pred:
            return images
        BG_derivative_label = batch["BG_derivative"].to(dtype=torch.long) - 1
        CSO_derivative_label = batch["CSO_derivative"].to(dtype=torch.long) - 1
        labels = [BG_derivative_label, CSO_derivative_label]
        return images, labels   