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

from bigdl.chronos.forecaster.tf import TCNForecaster
import pandas as pd
from bigdl.chronos.data import TSDataset, get_public_dataset
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.metric.forecast_metrics import Evaluator
import time
import numpy as np
import tensorflow as tf

look_back = 96
horizon = 720

from tensorflow.keras.utils import set_random_seed
set_random_seed(1)

tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

def generate_data():
    tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name='tsinghua_electricity',
                                                               with_split=True,
                                                               val_ratio=0.1,
                                                               test_ratio=0.2)
    standard_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.impute(mode="last")\
              .scale(standard_scaler, fit=(tsdata is tsdata_train))\
              .roll(lookback=look_back, horizon=horizon)
            
    train_loader = tsdata_train.to_tf_dataset(shuffle=True, batch_size=32)
    test_loader = tsdata_test.to_tf_dataset(batch_size=1)
    
    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = generate_data()
   
    forecaster = TCNForecaster(past_seq_len = look_back,
                               future_seq_len = horizon,
                               input_feature_num = 321,
                               output_feature_num = 321,
                               normalization=True)
    forecaster.fit(train_loader, epochs=30, batch_size=32) 
    

    metrics = forecaster.evaluate(test_loader, multioutput='uniform_average')
    print("MSE is:", metrics[0])

    latency = []
    for x, y in test_loader:
        x = x.numpy()
        st = time.time()
        yhat = forecaster.predict(x)
        latency.append(time.time()-st)
    # latency is calculated the mean after ruling out the first 10% and last 10%
    latency = latency[int(0.1*len(latency)):int(0.9*len(latency))]
    print("Inference latency(s) is:", np.mean(latency))

