#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import torch
from bigdl.chronos.pytorch import TSTrainer as Trainer
from bigdl.chronos.model.tcn import model_creator
from bigdl.chronos.metric.forecast_metrics import Evaluator
from bigdl.chronos.data.repo_dataset import get_public_dataset
from sklearn.preprocessing import StandardScaler

def gen_dataloader():
    tsdata_train, tsdata_val,\
        tsdata_test = get_public_dataset(name='nyc_taxi',
                                         with_split=True,
                                         val_ratio=0.1,
                                         test_ratio=0.1)

    stand = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
            .impute()\
            .gen_dt_feature()\
            .scale(stand, fit=tsdata is tsdata_train)\
            .roll(lookback=48,horizon=1)

    tsdata_traindataloader = tsdata_train.to_torch_data_loader(batch_size=32)
    tsdata_valdataloader = tsdata_val.to_torch_data_loader(batch_size=32, shuffle=False)
    tsdata_testdataloader = tsdata_test.to_torch_data_loader(batch_size=32, shuffle=False)

    return tsdata_traindataloader, tsdata_valdataloader, tsdata_testdataloader

def predict_wraper(model, input_sample):
    model(input_sample)

if __name__ == '__main__':

    # create data loaders for train/valid/test
    tsdata_traindataloader,\
    tsdata_valdataloader,\
    tsdata_testdataloader = gen_dataloader()
    
    # create a model
    # This could be an arbitrary model, we choose to use a built-in model TCN here
    config = {'input_feature_num':8,
              'output_feature_num':1,
              'past_seq_len':48,
              'future_seq_len':1,
              'kernel_size':3,
              'repo_initialization':True,
              'dropout':0.1,
              'seed': 0,
              'num_channels':[30]*7
              }
    model = model_creator(config)
    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
    lit_model = Trainer.compile(model, loss, optimizer)

    # train the model
    # You may use any method to train the model either on gpu or cpu
    trainer = Trainer(max_epochs=3,
                      accelerator='gpu', 
                      devices=1,
                     )
    trainer.fit(lit_model, tsdata_traindataloader, tsdata_testdataloader)

    # get an input sample
    x = None
    for x, _ in tsdata_traindataloader:
        break
    input_sample = x[0].unsqueeze(0)
    
    # speed up the model using Chronos TSTrainer
    speed_model = Trainer.trace(lit_model, accelerator="onnxruntime", input_sample=input_sample)

    # evaluate the model's latency
    print("original pytorch latency (ms):", Evaluator.get_latency(predict_wraper, lit_model, input_sample))
    print("onnxruntime latency (ms):", Evaluator.get_latency(predict_wraper, speed_model, input_sample))
