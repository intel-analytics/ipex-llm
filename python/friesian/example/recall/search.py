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
import argparse
import numpy as np
from bigdl.dllib.utils.log4Error import *
from bigdl.orca import init_orca_context, OrcaContext, stop_orca_context
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, StringType
from utils import *


def faiss_search(faiss_index_pkl, item_dict_pkl, cluster_mode, batch_size=65536, k=200):
    import pickle
    import faiss

    if cluster_mode == 'yarn':
        load_SPARK = True
    else:
        load_SPARK = False

    def do_search(partition):
        with open_pickle(faiss_index_pkl, load_SPARK) as index_pkl:
            faiss_idx = faiss.deserialize_index(pickle.load(index_pkl))

        with open_pickle(item_dict_pkl, load_SPARK) as f:
            item_dict = pickle.load(f)

        buffer = []
        for record in partition:
            if len(buffer) == batch_size:
                s1 = time.time()
                seed_ids = [row[0] for row in buffer]
                embeddings = [row[1] for row in buffer]
                buffer = [record]
                q_vec = np.stack(embeddings).astype(np.float32)
                similarity_array, idx_array = faiss_idx.search(q_vec, k=k)
                e1 = time.time()
                print("Search time: ", e1 - s1)

                for i in range(batch_size):
                    seed_idx = int(seed_ids[i])
                    seed_item = str(item_dict[seed_idx])
                    for n, (score, rec_id) in enumerate(
                            zip(similarity_array[i], idx_array[i])
                    ):
                        rec_id = int(rec_id)
                        yield (seed_item, str(item_dict[rec_id]), int(n), float(score))

            else:
                buffer.append(record)

        remain_size = len(buffer)
        if remain_size > 0:
            seed_ids = [row[0] for row in buffer]
            embeddings = [row[1] for row in buffer]
            q_vec = np.stack(embeddings).astype(np.float32)
            similarity_array, idx_array = faiss_idx.search(q_vec, k=k)

            for i in range(remain_size):
                seed_idx = int(seed_ids[i])
                seed_item = str(item_dict[seed_idx])
                for n, (score, rec_id) in enumerate(
                        zip(similarity_array[i][1:], idx_array[i][1:])
                ):
                    rec_id = int(rec_id)
                    yield (seed_item, str(item_dict[rec_id]), int(n + 1), float(score))

    return do_search


def search(args):
    set_env(args.num_threads)
    if args.cluster_mode == "yarn":
        num_executors = 12
        executor_cores = args.num_threads
        executor_memory = "12g"
        driver_cores = 4
        driver_memory = "4g"
        sc = init_orca_context("yarn", cores=executor_cores,
                               num_nodes=num_executors, memory=executor_memory,
                               driver_cores=driver_cores, driver_memory=driver_memory,
                               extra_python_lib="utils.py")

        print('add files to spark >>>>>>')
        sc.addFile(args.dict_path)
        sc.addFile(args.faiss_index_path)

        args.faiss_index_path = args.faiss_index_path.split('/')[-1]
        args.dict_path = args.dict_path.split('/')[-1]

    elif args.cluster_mode == "local":
        sc = init_orca_context("local", cores=8)
        num_executors = 4
    else:
        invalidInputError(False, "cluster_mode should be one of 'local', "
                                 "'yarn', but got " + args.cluster_mode)
        sys.exit()

    spark = OrcaContext.get_spark_session()
    with StopWatch("do_search spark >>>>>>") as sw:
        df = spark.read.parquet(args.parquet_path)
        print('Total number of items: ', df.count())
        rdd = df.rdd.repartition(num_executors)  # Each node runs one faiss task
        res_rdd = rdd.mapPartitions(
            faiss_search(args.faiss_index_path, args.dict_path,
                         args.cluster_mode, batch_size=args.batch_size,
                         k=args.top_k))
        schema = StructType([
            StructField('seed_item', StringType(), False),
            StructField('rec_item', StringType(), False),
            StructField('rank', IntegerType(), False),
            StructField('similarity_score', FloatType(), False)
        ])
        res_df = spark.createDataFrame(res_rdd, schema=schema)
        res_df.write.mode("overwrite").parquet(args.parquet_output_path)
        stop_orca_context()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Set parameters for search')

    parser.add_argument('--num_threads', type=int, default=8,
                        help='Set the environment variable OMP_NUM_THREADS for each faiss task')
    parser.add_argument('--cluster_mode', type=str, default='yarn',
                        help='The cluster mode, such as local, yarn')

    parser.add_argument('--dict_path', type=str, default='./item_dict.pkl',
                        help='Path to item_dict.pkl')
    parser.add_argument('--faiss_index_path', type=str,
                        default='./index_FlatL2.pkl',
                        help='Path to faiss index data path')
    parser.add_argument('--parquet_path', type=str, default='./data.parquet',
                        help='Path to input parquet data (query items)')
    parser.add_argument('--parquet_output_path', type=str,
                        default='./similarity_search_L2.parquet',
                        help='Path to save output parquet date (search results)')

    parser.add_argument('--top_k', type=int, default=100,
                        help='Number of items to be searched for each query item')
    parser.add_argument('--batch_size', type=int, default=50000,
                        help='Batch size for each faiss task')

    args = parser.parse_args()
    search(args)
