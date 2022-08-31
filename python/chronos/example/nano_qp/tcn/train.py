from tcn_config import *
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.cli import LightningCLI
import pytorch_lightning as pl
from model import LitTCN
from data import gen_dataloader
import time
import torch
import matplotlib.pyplot as plt
    
if __name__ == '__main__':
    tsdata_traindataloader,\
    tsdata_valdataloader,\
    tsdata_testdataloader,\
    tsdata_test = gen_dataloader()
    
    config = {'input_feature_num':input_feature_num,
              'output_feature_num':output_feature_num,
              'past_seq_len':past_seq_len,
              'future_seq_len':feature_seq_len,
              'kernel_size':3,
              'repo_initialization':True,
              'dropout':0.1,
              'seed': seed,
              'num_channels':[30]*7
              }
    
    checkpoint_callback = ModelCheckpoint(
                            monitor='val loss',
                            filename='tcn',
                            save_top_k=1,
                            mode='min',
                            # save_weights_only = True
                            )
    
    model = LitTCN(config)
    trainer = pl.Trainer(max_epochs=3, val_check_interval=1.0,\
            #   gpus=0, 
              callbacks=[checkpoint_callback],
              accelerator='gpu', 
              devices=1,
              )
    trainer.fit(model, tsdata_traindataloader, tsdata_testdataloader)
    
    start = time.time()
    outputs = trainer.predict(model, tsdata_testdataloader)
    print(f'original_time:{time.time() - start}s')

    gt = [output['gt'] for output in outputs]
    pred = [output['pred'] for output in outputs]
    gt = torch.cat(gt) 
    pred = torch.cat(pred)
    pred_unscale = tsdata_test.unscale_numpy(pred)
    groundtruth_unscale = tsdata_test.unscale_numpy(gt)
    
    plt.figure(figsize=(24,6))
    plt.plot(pred_unscale[:,:,0])
    plt.plot(groundtruth_unscale[:,:,0])
    plt.legend(["prediction", "ground truth"])
    plt.savefig(f'./img/horizon:{horizon}_lookback:\
                {lookback}_lr:{lr}_epoch:{num_epochs}.png')