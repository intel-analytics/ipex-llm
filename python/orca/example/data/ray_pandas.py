#
# Copyright 2018 Analytics Zoo Authors.
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

import sys
from optparse import OptionParser

import pandas as pd
from bigdl.util.common import get_node_and_core_number
from pyspark.sql import SQLContext

import zoo.orca.data.pandas
from zoo import init_spark_on_local
from zoo.ray import RayContext


def process_feature(df, awake_begin=6, awake_end=23):
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df['hours'] = df['datetime'].dt.hour
    df['awake'] = (((df['hours'] >= awake_begin) & (df['hours'] <= awake_end))
                   | (df['hours'] == 0)).astype(int)
    return df


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-f", type=str, dest="file_path",
                      help="The file path to be read")

    (options, args) = parser.parse_args(sys.argv)

    # Prepare csv files
    df = pd.read_csv(options.file_path)
    sc = init_spark_on_local(cores="*")
    sqlContext = SQLContext(sc)
    num_nodes, num_cores = get_node_and_core_number()
    df_spark = sqlContext.createDataFrame(df)
    df_spark.printSchema()
    df_spark.repartition(num_cores).write.\
        format('json').mode("overwrite").save("/tmp/ray-pandas-example")

    # init ray context
    ray_ctx = RayContext(sc=sc,
                         object_store_memory="5g"
                         )
    ray_ctx.init()

    # read data
    data_shard = zoo.orca.data.pandas.read_json("/tmp/ray-pandas-example", ray_ctx,
                                                orient='columns', lines=True)

    # collect data
    data = data_shard.collect()
    print("collected data :")
    print(data[0].head())

    # repartition
    partitions = data_shard.get_partitions()
    print("get %d partitions" % len(partitions))
    data_shard.repartition(2)
    new_partitions = data_shard.get_partitions()
    print("get %d partitions after repartition" % len(new_partitions))

    # apply function on each element
    data_shards = data_shard.transform_shard(process_feature, 6, 24)
    data2 = data_shards.collect()
    print("collected new data :")
    print(data2[0].head())

    ray_ctx.stop()
    sc.stop()
