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

from bigdl.chronos.forecaster.autoformer_forecaster import AutoformerForecaster
from bigdl.chronos.data import TSDataset

import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import numpy as np
import torch


look_back = 96
horizon = 720
label_len = 48

def generate_data():
    # read the data
    raw_df = pd.read_csv('electricity.csv', parse_dates=["date"])
    target = []
    for i in range(0, 320):
        target.append(str(i))
    target.append("OT")

    # use TSDataset to split and preprocess the data
    tsdata_train, _, tsdata_test =\
        TSDataset.from_pandas(raw_df, dt_col="date", target_col=target, with_split=True, test_ratio=0.2, val_ratio=0.1)
    standard_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.impute()\
              .scale(standard_scaler, fit=(tsdata is tsdata_train))

    # get dataset for training, evaluation and prediction
    train_loader = tsdata_train.to_torch_data_loader(lookback=look_back, horizon=horizon,
                                                     time_enc=True, label_len=label_len)
    test_loader = tsdata_test.to_torch_data_loader(lookback=look_back, horizon=horizon,
                                                   time_enc=True, label_len=label_len, shuffle=False)
    pred_loader = tsdata_test.to_torch_data_loader(lookback=look_back, horizon=horizon,
                                                   time_enc=True, label_len=label_len, is_predict=True,
                                                   shuffle=False, batch_size=1)

    return train_loader, test_loader, pred_loader


if __name__ == '__main__':
    # init the forecaster
    forecaster = AutoformerForecaster(past_seq_len=look_back,
                                      future_seq_len=horizon,
                                      input_feature_num=321,
                                      output_feature_num=321,
                                      label_len=label_len,
                                      freq='h',
                                      seed=2)  # 

    # get data
    train_loader, test_loader, pred_loader = generate_data()

    # # fit the model
    forecaster.fit(train_loader, epochs=3, batch_size=32)

    # # save the model
    ts = time.time()
    forecaster.save(f"autoformer_{look_back}_{horizon}_321_epoch_3_{ts}.fxt")
    forecaster.load(f"autoformer_{look_back}_{horizon}_321_epoch_3_{ts}.fxt")

    # evaluate on test set
    evaluate = forecaster.evaluate(test_loader)
    print("MSE on test dataset:", evaluate)

    # thread=8, latency
    forecaster.internal.eval()
    time_list = []
    torch.set_num_threads(8)
    with torch.no_grad():
        for i, batch in enumerate(pred_loader):
            st = time.time()
            forecaster.internal.predict_step(batch, i)
            time_list.append(time.time() - st)
    # latency is calculated the mean after ruling out the first 10% and last 10%
    time_list = time_list[int(0.1*len(time_list)):int(0.9*len(time_list))]
    print("latency(8 cores):", np.mean(time_list))

    # thread=1, latency
    forecaster.internal.eval()
    time_list = []
    torch.set_num_threads(1)
    with torch.no_grad():
        for i, batch in enumerate(pred_loader):
            st = time.time()
            forecaster.internal.predict_step(batch, i)
            time_list.append(time.time() - st)
    # latency is calculated the mean after ruling out the first 10% and last 10%
    time_list = time_list[int(0.1*len(time_list)):int(0.9*len(time_list))]
    print("latency(1 cores):", np.mean(time_list))
