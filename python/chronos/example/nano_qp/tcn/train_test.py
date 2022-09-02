from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from model import LitTCN
import time
import torch
import matplotlib.pyplot as plt
from bigdl.nano.pytorch import Trainer
from bigdl.chronos.model.tcn import model_creator
from bigdl.chronos.metric.forecast_metrics import Evaluator
from tcn_config import *
from bigdl.chronos.data.repo_dataset import get_public_dataset
from sklearn.preprocessing import StandardScaler


lookback = 48 # number of steps to look back
horizon = 1 # number of steps to predict
input_feature_num = 8 # number of feature to use
output_feature_num = horizon
past_seq_len = lookback
feature_seq_len = 1 # number of feature to predict
optimizer = 'Adam'
lr = 0.001
batch_size = 32
num_epochs = 3
seed = 0
device = "gpu"
save_dir = './model'
save_model_file = f'horizon:{horizon}lookback:{lookback}lr:{lr}_epoch:{num_epochs}' + 'tcn.pth'

def gen_dataloader():
    tsdata_train, tsdata_val,\
        tsdata_test = get_public_dataset(name='nyc_taxi',
                                        with_split=True,
                                        val_ratio=0.1,
                                        test_ratio=0.1
                                        )
    # carry out additional customized preprocessing on the dataset.
    stand = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
            .impute()\
            .gen_dt_feature()\
            .scale(stand, fit=tsdata is tsdata_train)\
            .roll(lookback=lookback,horizon=horizon)

    tsdata_traindataloader = tsdata_train.to_torch_data_loader(batch_size=batch_size)
    tsdata_valdataloader = tsdata_val.to_torch_data_loader(batch_size=batch_size, shuffle=False)
    tsdata_testdataloader = tsdata_test.to_torch_data_loader(batch_size=batch_size, shuffle=False)

    return tsdata_traindataloader,\
           tsdata_valdataloader,\
           tsdata_testdataloader,\
           tsdata_test 

def test_time(model, dataloader):
    for x, _ in dataloader:
        res = model(x[0].unsqueeze(0))
        break
    return res

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
    model = model_creator(config)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=lr, params=model.parameters())
    lit_model = Trainer.compile(model, loss, optimizer)
    trainer = pl.Trainer(max_epochs=3, val_check_interval=1.0,
              accelerator='gpu', 
              devices=1,
              )
    trainer.fit(lit_model, tsdata_traindataloader, tsdata_testdataloader)
    
    
    print("original pytorch runtime (ms):", Evaluator.get_latency(test_time,lit_model, tsdata_testdataloader))
    
    speed_model = Trainer.trace(lit_model, accelerator="onnxruntime", input_sample=tsdata_testdataloader)
    print("onnxruntime pytorch runtime (ms):", Evaluator.get_latency(test_time, speed_model, tsdata_testdataloader))