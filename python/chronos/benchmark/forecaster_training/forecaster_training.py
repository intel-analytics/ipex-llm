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


import time
import json
import argparse
from bigdl.chronos.forecaster import TCNForecaster
from bigdl.chronos.data import get_public_dataset
from sklearn.preprocessing import StandardScaler

lookback, horizon = 48, 1

parser = argparse.ArgumentParser(description="TCNForecaster Training")
parser.add_argument("--name", default="TCNForecaster Training Baseline", type=str)

def create_data():
    tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name="nyc_taxi")

    scaler = StandardScaler()

    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
              .impute()\
              .gen_dt_feature()\
              .scale(scaler, fit=(tsdata is tsdata_train))\
              .roll(lookback=lookback, horizon=horizon)
    
    return tsdata_train, tsdata_val
    


def main():
    args = parser.parse_args()
    train_data, val_data = create_data()
    x, y = train_data.to_numpy()
    forecaster = TCNForecaster(past_seq_len=lookback,
                               future_seq_len=horizon,
                               input_feature_num=x.shape[-1],
                               output_feature_num=y.shape[-1],
                               seed=1)
    train_start = time.time()
    forecaster.fit((x,y), validation_data=val_data, epochs=3)
    train_end = time.time()

    output = json.dumps({
        "config": args.name,
        "train_time": train_end - train_start
    })

    print(f'>>>{output}<<<')

if __name__ == "__main__":
    main()