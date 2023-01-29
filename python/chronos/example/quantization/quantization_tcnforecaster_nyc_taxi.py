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

from bigdl.chronos.data import get_public_dataset
from bigdl.chronos.forecaster import TCNForecaster
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.metric.forecast_metrics import Evaluator
import time
import numpy as np


def get_tsdata(lookback=96, horizon=320):
    name = 'nyc_taxi'
    tsdata_train, _, tsdata_test = get_public_dataset(name, val_ratio=0, test_ratio=0.4)
    stand_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.gen_dt_feature(features=["HOUR"], one_hot_features=["HOUR"])\
              .impute(mode="linear")\
              .scale(stand_scaler, fit=(tsdata is tsdata_train))\
              .roll(lookback=lookback, horizon=horizon)
    return tsdata_train, tsdata_test


if __name__ == "__main__":
    tsdata_train, tsdata_test = get_tsdata()
    x_train, y_train = tsdata_train.to_numpy()
    x_test, y_test = tsdata_test.to_numpy()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    forecaster = TCNForecaster(past_seq_len = 96,
                               future_seq_len = 320,
                               input_feature_num = x_train.shape[-1],
                               output_feature_num = 1,
                               normalization=True,
                               num_channels = [48] * 7)
    forecaster.fit((x_train, y_train), epochs=1, batch_size=32)
    forecaster.num_processes = 1

    st = time.time()
    y_pred = forecaster.predict(x_test, batch_size=256)
    fp32_pytorch_time = time.time()-st

    y_pred_unscale = tsdata_test.unscale_numpy(y_pred)
    y_test_unscale = tsdata_test.unscale_numpy(y_test)
    avg_smape_fp32_pytorch = Evaluator.evaluate("smape", y_test_unscale, y_pred_unscale, aggregate='mean')[0]

    forecaster.quantize((x_train, y_train), framework='pytorch_fx')
    st = time.time()
    y_pred = forecaster.predict(x_test, quantize=True, batch_size=256)
    int8_pytorch_time = time.time()-st

    forecaster.quantize((x_train, y_train), framework='onnxrt_qlinearops')
    y_pred_unscale = tsdata_test.unscale_numpy(y_pred)
    y_test_unscale = tsdata_test.unscale_numpy(y_test)
    avgr_smape_int8_pytorch = Evaluator.evaluate("smape", y_test_unscale, y_pred_unscale, aggregate='mean')[0]

    y_pred = forecaster.predict_with_onnx(x_test, quantize=True)
    y_pred_unscale = tsdata_test.unscale_numpy(y_pred)
    y_test_unscale = tsdata_test.unscale_numpy(y_test)
    avgr_smape_int8_onnx = Evaluator.evaluate("smape", y_test_unscale, y_pred_unscale, aggregate='mean')[0]

    fp32_pytorch_latency = []
    for i in range(x_test.shape[0]):
        x = x_test[i:i+1]
        st = time.time()
        y_pred = forecaster.predict(x, acceleration=False)
        fp32_pytorch_latency.append(time.time()-st)

    int8_onnx_latency = []
    for i in range(x_test.shape[0]):
        x = x_test[i:i+1]
        st = time.time()
        y_pred = forecaster.predict_with_onnx(x, quantize=True)
        int8_onnx_latency.append(time.time()-st)

    print("Pytorch Quantization helps increase inference throughput by", round(fp32_pytorch_time/int8_pytorch_time*100-100, 2), "%")
    print("Onnx Quantization helps decrease inference latency by", round((np.median(fp32_pytorch_latency)-np.median(int8_onnx_latency))/np.median(fp32_pytorch_latency)*100, 2), "%")
    print("fp32 pytorch smape:", round(float(avg_smape_fp32_pytorch), 2))
    print("int8 pytorch smape:", round(float(avgr_smape_int8_pytorch), 2))
    print("int8 onnx smape:", round(float(avgr_smape_int8_onnx), 2))
