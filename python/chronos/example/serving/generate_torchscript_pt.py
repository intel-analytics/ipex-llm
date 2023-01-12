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
from bigdl.chronos.data import get_public_dataset
from sklearn.preprocessing import StandardScaler
from bigdl.chronos.pytorch import TSInferenceOptimizer as InferenceOptimizer
import torch

if __name__ == "__main__":
    scaler = StandardScaler()

    # use nyc_taxi public dataset
    train_data, val_data, test_data = get_public_dataset("nyc_taxi",
                                                         with_split=True, test_ratio=0.1)
    for data in [train_data, val_data, test_data]:
        data.scale(scaler, fit=data is train_data).roll(lookback=6, horizon=2)

    # create a forecaster
    forecaster = TCNForecaster.from_tsdataset(train_data)

    # train the forecaster
    forecaster.fit(train_data, epochs=10)

    model = InferenceOptimizer.trace(forecaster.internal, accelerator="jit",
                                     input_sample=torch.randn(1, 6, 1))

    InferenceOptimizer.save(model, path="checkpoint")
