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
from bigdl.chronos.data import get_public_dataset
from sklearn.preprocessing import StandardScaler

lookback = 48
horizon = 1

parser = argparse.ArgumentParser(description="TSDataset processing")
parser.add_argument("--name", default="nyc_taxi baseline", type=str)

def data_processing():
    tsdata_train, tsdata_val, tsdata_test = get_public_dataset(name="nyc_taxi")

    scaler = StandardScaler()

    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.deduplicate()\
              .impute()\
              .gen_dt_feature()\
              .scale(scaler, fit=(tsdata is tsdata_train))\
              .roll(lookback=lookback, horizon=horizon)


if __name__ == "__main__":
    args = parser.parse_args()

    process_start = time.time()
    data_processing()
    process_end = time.time()

    output = json.dumps({
        "config": args.name,
        "process_time": process_end - process_start
    })

    print(f'>>>{output}<<<')
