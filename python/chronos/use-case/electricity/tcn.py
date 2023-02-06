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

from bigdl.chronos.forecaster import TCNForecaster
import pandas as pd
from bigdl.chronos.data import TSDataset, get_public_dataset
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.metric.forecast_metrics import Evaluator
import torch
import time
import numpy as np

look_back = 96
horizon = 720

def generate_data():
    tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='tsinghua_electricity',
                                                               with_split=True,
                                                               val_ratio=0.1,
                                                               test_ratio=0.2)
    standard_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.impute(mode="last")\
              .scale(standard_scaler, fit=(tsdata is tsdata_train))
            
    train_loader = tsdata_train.to_torch_data_loader(lookback=look_back, horizon=horizon)
    test_loader = tsdata_test.to_torch_data_loader(batch_size=1, lookback=look_back, horizon=horizon, shuffle=False)
    
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = generate_data()
   
    forecaster = TCNForecaster(past_seq_len = look_back,
                               future_seq_len = horizon,
                               input_feature_num = 321,
                               output_feature_num = 321,
                               dummy_encoder=True,
                               num_channels = [30] * 7,
                               repo_initialization = False,
                               kernel_size = 3, 
                               dropout = 0.1, 
                               lr = 0.001,
                               seed = 1)
    forecaster.num_processes = 1
    forecaster.fit(train_loader, epochs=30, batch_size=32) 
    

    metrics = forecaster.evaluate(test_loader, multioutput='uniform_average')
    print("MSE is:", metrics[0])

    torch.set_num_threads(1)
    latency = []
    for x, y in test_loader:
        st = time.time()
        yhat = forecaster.predict(x.numpy())
        latency.append(time.time()-st)
    # latency is calculated the mean after ruling out the first 10% and last 10%
    latency = latency[int(0.1*len(latency)):int(0.9*len(latency))]
    print("Inference latency is:", np.mean(latency))

    forecaster.build_onnx(thread_num=1)
    onnx_latency = []
    for x, y in test_loader:
        st = time.time()
        y_pred = forecaster.predict_with_onnx(x.numpy())
        onnx_latency.append(time.time()-st)
    # latency is calculated the mean after ruling out the first 10% and last 10%
    onnx_latency = onnx_latency[int(0.1*len(onnx_latency)):int(0.9*len(onnx_latency))]
    print("Inference latency with onnx is:", np.mean(onnx_latency))
