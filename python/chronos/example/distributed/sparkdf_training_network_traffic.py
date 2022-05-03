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
from bigdl.orca.common import init_orca_context, stop_orca_context, OrcaContext
from bigdl.chronos.data.experimental import XShardsTSDataset
from bigdl.chronos.forecaster import TCNForecaster
import numpy as np

def generate_spark_df():
    sc = OrcaContext.get_spark_context()
    spark = OrcaContext.get_spark_session()
    df = spark.read.format("csv")\
                   .option("inferSchema", "true")\
                   .option("header", "true")\
                   .load("/home/junweid/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv")
    tsdata_train, _, tsdata_test = XShardsTSDataset.from_sparkdf(df, dt_col="timestamp",
                                               target_col=["value"],
                                               with_split=True,
                                               val_ratio=0,
                                               test_ratio=0.1)
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.roll(lookback=100, horizon=10)
    return tsdata_train, tsdata_test

if __name__ == '__main__':
    init_orca_context(cores=8)
    tsdata_train, tsdata_test = generate_spark_df()
    data_train = tsdata_train.to_xshards()
    data_test = tsdata_test.to_xshards()

    forecaster = TCNForecaster(past_seq_len=100,
                               future_seq_len=10,
                               input_feature_num=1,
                               output_feature_num=1,
                               metrics=['mse', 'mae'],
                               distributed=True,
                               workers_per_node=1,
                               seed=0)

    forecaster.fit(data_train, epochs=20, batch_size=512)

    evaluate_result = forecaster.evaluate(data_test, multioutput="uniform_average")
    print(evaluate_result)

    stop_orca_context()
