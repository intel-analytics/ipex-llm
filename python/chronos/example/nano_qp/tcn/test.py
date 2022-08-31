from tcn_config import *
from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.cli import LightningCLI
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch
import time
from model import LitTCN
from data import gen_dataloader


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
    
    model = LitTCN(config)
    model = model.load_from_checkpoint('./plcode/lightning_logs/version_0/checkpoints/tcn.ckpt')
    speed_model = LitTCN(config)
    speed_model = speed_model.speed('onnxruntime', tsdata_testdataloader)
    model = speed_model.load_from_checkpoint('./lightning_logs/version_0/checkpoints/tcn.ckpt')
    trainer = pl.Trainer()
    # start = time.time()
    # outputs = trainer.predict(model, tsdata_testdataloader)
    # print(f'original_time:{time.time() - start}s')
    
    start = time.time()
    # outputs = trainer.predict(speed_model, tsdata_testdataloader, ckpt_path='./lightning_logs/version_14/checkpoints/tcn.ckpt')
    outputs = trainer.predict(speed_model, tsdata_testdataloader)
    print(f'onnx_speed_time:{time.time() - start}s')

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
    plt.savefig(f'./img/result.png')
    