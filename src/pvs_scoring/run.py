import os
# os.environ["OMP_NUM_THREADS"] = "1"

import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI
from torch.optim.lr_scheduler import PolynomialLR
from omegaconf import OmegaConf
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    cli = CustomLightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        parser_kwargs={"parser_mode": "omegaconf"},
    )
    
    
class CustomLightningCLI(LightningCLI):

    @staticmethod
    def configure_optimizers(lightning_module, optimizer, lr_scheduler=None):

        if lr_scheduler is None:
            return optimizer
        
        if isinstance(lr_scheduler, PolynomialLR):
            total_iters = lightning_module.trainer.max_epochs
            lr_scheduler.total_iters = total_iters
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
            }
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "epoch"},
        }


def length(lst: list) -> int:
    return len(lst)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("length", length)


if __name__ == "__main__":
    main()