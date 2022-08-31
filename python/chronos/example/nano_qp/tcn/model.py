from bigdl.chronos.model.tcn import model_creator
from tcn_config import *
# from pytorch_lightning.utilities.cli import LightningCLI
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from bigdl.nano.pytorch import Trainer

class LitTCN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = model_creator(config)
        
    def speed(self, mode, input_sample):
        self.model = Trainer.trace(self.model, accelerator=mode, input_sample=input_sample)
        
        return self
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        return {'gt':y, 'pred':y_hat}

    def _shared_step(self, batch, mode):
        x, y = batch
        y_hat = self.model(x)
        loss = F.mse_loss(y_hat, y)
        self.log(mode + ' loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(lr=lr, params=model.parameters())