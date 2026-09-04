import pytorch_lightning as pl
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any

class Base(pl.LightningModule):
    
    def __init__(
        self,
        network: nn.Module,
        loss_fn: nn.Module | list[nn.Module],
        metric: Any,
        watch_log_freq: int = 100,
        dir_pred: str | None = None
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(ignore=["network"])
        self.network = network
        self.criterion =  loss_fn
        self.metric = metric
        self.watch_log_freq = watch_log_freq
        self.dir_pred = Path(dir_pred) if dir_pred is not None else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    # def on_train_start(self) -> None:
    #     self.logger.watch(self.network, log="all", log_freq=self.watch_log_freq)
     
    def on_validation_epoch_end(self) -> None:
        metric_dict = self.metric.aggregate()
        self.metric.reset()
        self.log_dict(metric_dict, on_step=False, on_epoch=True)
         
    def on_predict_start(self):
        self.dir_pred.mkdir(parents=True, exist_ok=False)
        
    def on_predict_end(self):
        (Path.cwd() / "config.yaml").unlink()
    
    def training_step(self, batch: dict) -> torch.Tensor:
        raise NotImplementedError
    
    def validation_step(self, batch: dict) -> None:
        raise NotImplementedError
    
    def predict_step(self, batch: dict) -> None:
        raise NotImplementedError