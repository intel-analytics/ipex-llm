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

import sys
import time

from tpch_function import *
from pyspark.sql import SparkSession


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("\t<spark-submit> --master local main.py")
        print("\t\t<tpch_data_root_path> <query_number> <num_iterations> <true for SQL | false for functional>")
        exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    query_list = []
    if (len(sys.argv) > 3):
        query_number = int(sys.argv[3])
        query_list.append(query_number)
    else:
        query_list = range(1, 23)

    spark = SparkSession.builder.getOrCreate()
    print("Will run " + str(query_list))
    start1 = time.time()
    queries = TpchFunctionalQueries(spark, input_dir)
    for iter in query_list:
        query_number = iter
        print("TPCH Starting query #{0}".format(iter))
        start = time.time()
        out = getattr(queries, "q" + str(query_number))()
        out.write.mode("overwrite").format("csv").option("header", "true").save(output_dir + "./Q" + str(query_number))
        end = time.time()
        print("query%s --finished--,time is %s s" % (query_number, end - start))
    end1 = time.time()
    print("total time is %f s" % (end1 - start1))

    spark.stop()


if __name__ == '__main__':
    main()
