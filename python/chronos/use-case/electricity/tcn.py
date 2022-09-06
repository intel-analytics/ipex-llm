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

from bigdl.chronos.forecaster.tcn_forecaster import TCNForecaster
import pandas as pd
from bigdl.chronos.data import TSDataset
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.metric.forecast_metrics import Evaluator
from torch.utils.data.dataloader import DataLoader
import torch
import time
import numpy as np

look_back = 96
horizon = 720

def generate_data():
    raw_df = pd.read_csv("electricity.csv")
    df = pd.DataFrame(pd.to_datetime(raw_df.date))
    for i in range(0, 320):
        df[str(i)] = raw_df[str(i)]
    df["OT"] = raw_df["OT"]

    target = []
    for i in range(0, 320):
        target.append(str(i))
    target.append("OT")

    tsdata_train, tsdata_val, tsdata_test = TSDataset.from_pandas(df, dt_col="date", target_col=target,
                                                                  with_split=True, test_ratio=0.2, val_ratio=0.1)
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
                            num_channels = [30] * 7,
                            repo_initialization = False,
                            kernel_size = 3, 
                            dropout = 0.1, 
                            lr = 0.001,
                            seed = 1)
    forecaster.num_processes = 1
    forecaster.fit(train_loader, epochs=30, batch_size=32) 
    

    metric = []
    for x, y in test_loader:
        yhat = forecaster.predict(x.numpy())
        metric.append(Evaluator.evaluate("mse", y.detach().numpy(), yhat))
    print("MSE is:", np.mean(metric))

    torch.set_num_threads(1)
    latency = []
    with torch.no_grad():
        for x, y in test_loader:
            st = time.time()
            yhat = forecaster.predict(x.numpy())
            latency.append(time.time()-st)
    print("Inference latency is:", np.median(latency))

    forecaster.build_onnx(thread_num=1)
    onnx_latency = []
    with torch.no_grad():
        for x, y in test_loader:
            st = time.time()
            y_pred = forecaster.predict_with_onnx(x.numpy())
            onnx_latency.append(time.time()-st)
    print("Inference latency with onnx is:", np.median(onnx_latency))
