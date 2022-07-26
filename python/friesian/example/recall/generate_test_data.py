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

import pickle
import string
import numpy as np
import math
import faiss
import sys
import random
import argparse


def create_dummy_data(row_nums, vec_dim, verbose, rnd_seed=42, HEADER_LEN=8, only_item=False):
    # ref: https://codereview.stackexchange.com/questions/151284/random-string-generating-function
    _CHARACTERS = np.array(list(string.ascii_uppercase + string.digits))
    _ITEM_HEADER_LEN = HEADER_LEN

    def _dummy_name(rnd_gen, index, index_len):
        header = "".join(rnd_gen.choice(_CHARACTERS, size=_ITEM_HEADER_LEN))
        return header + "_" + str(index).zfill(index_len)

    str_len = len(str(row_nums))
    rng = np.random.default_rng(rnd_seed)  # reset
    item_dict = np.array([_dummy_name(rng, i, str_len) for i in range(row_nums)]).copy(
        order="C"
    )
    if only_item:
        return item_dict
    else:
        rng = np.random.default_rng(rnd_seed)
        emb_vecs = (
            (rng.random((vec_dim, row_nums)) + rng.random(row_nums))
            .astype(np.float32)
            .transpose()
            .copy(order="C")
        )
        if verbose:
            print('emb_vecs shape: {}, item_dict shape: {}'.format(emb_vecs.shape, item_dict.shape))
        return emb_vecs, item_dict


def search_sample(index, sample_vec):
    print("search sample test >>>>>>")
    similarity_array, idx_array = index.search(sample_vec, k=10)
    print(idx_array, similarity_array)


def gen_vector(x, vec_dim):
    np.random.seed(x)  # make sure get same vectors for each specific index
    return (x, np.random.rand(vec_dim).astype(np.float32) +
            random.randint(0, 1000) * random.random())


def generate_data(args):
    print("create emb_vecs and item_dict data >>>>>>")
    from bigdl.orca import init_orca_context, OrcaContext, stop_orca_context
    from pyspark.sql.types import Row
    from pyspark.sql.types import StructType, StructField, IntegerType, ArrayType, FloatType

    print('>>>>>> using spark >>>>>>')
    sc = init_orca_context(cores="*", memory="16g", conf={"spark.driver.maxResultSize": "10g"})
    spark = OrcaContext.get_spark_session()
    rdd = sc.parallelize(range(args.row_nums))
    rdd = rdd.map(lambda row: gen_vector(row, args.vec_dim))
    rdd.cache()  # If not cache, will run random generation again when saving to parquet.
    data = rdd.collect()
    emb_vecs = np.array([row[1] for row in data], dtype=np.float32)
    print('vector shape: ', emb_vecs.shape)
    # np.save("data.npy", vectors)
    schema = StructType([
        StructField('id', IntegerType(), False),
        StructField('embedding', ArrayType(FloatType()), False)
    ])
    df = spark.createDataFrame(rdd.map(lambda x: Row(x[0], x[1].tolist())), schema=schema)
    df.write.mode("overwrite").parquet(args.parquet_path)
    print("Finished")
    stop_orca_context()
    item_dict = create_dummy_data(args.row_nums, args.vec_dim, args.verbose, rnd_seed=42,
                                  HEADER_LEN=args.header_len, only_item=True)

    print("create index for faiss >>>>>>")
    if args.index_type == 'FlatL2':
        print('index_type: ', "# FlatL2")
        index_faiss = faiss.IndexFlatL2(args.vec_dim)
        # index_flat = faiss_flat.IndexFlatIP(VEC_DIM)
        print('index_flat.is_trained: {}'.format(index_faiss.is_trained))
        index_faiss.add(emb_vecs)
        print('index_faiss.ntotal: ', index_faiss.ntotal)
    elif args.index_type == 'IVFFlatL2':
        print('index_type: ', "# IVFFlatL2")
        # The number of cells (space partition). Typical value is sqrt(N)
        NLIST = int(math.sqrt(args.row_nums))
        print('NLIST: ', NLIST)
        quantizer = faiss.IndexFlatL2(args.vec_dim)
        index_faiss = faiss.IndexIVFFlat(quantizer, args.vec_dim, NLIST)
        print('before, index_flat.is_trained: {}'.format(index_faiss.is_trained))
        index_faiss.train(emb_vecs)
        print('after, index_flat.is_trained: {}'.format(index_faiss.is_trained))
        index_faiss.add(emb_vecs)
    else:
        print('unsupported index_type input')
        sys.exit()

    q_vec = emb_vecs[[0, ]]
    search_sample(index_faiss, q_vec)

    print("saving created data and index >>>>>>")
    for file, obj in [(args.emb_path, emb_vecs), (args.dict_path, item_dict)]:
        with open(file, "wb") as f:
            print('saving to: {}'.format(file))
            pickle.dump(obj, f)
    # https://github.com/matsui528/faiss_tips#io

    with open(args.index_save_path, "wb") as f:
        print('saving to: {}'.format(args.index_save_path))
        # serialize the index into binary array (np.array).
        # You can save/load it via numpy IO functions.
        chunk = faiss.serialize_index(index_faiss)
        pickle.dump(chunk, f)

    if args.verbose:
        # sample
        print('start loading data for testing >>>>>>')
        print(emb_vecs.shape)
        print(emb_vecs[0, :10])
        print(item_dict.shape)
        print(item_dict[:10])

        print(f"loading {args.emb_path}, {args.dict_path}")

        with open(args.emb_path, "rb") as f:
            emb_vecs_loaded = pickle.load(f)
        print(emb_vecs_loaded.shape)
        print(emb_vecs_loaded[0, :10])

        with open(args.dict_path, "rb") as f:
            item_dict_loaded = pickle.load(f)
        print(item_dict_loaded.shape)
        print(item_dict_loaded[:10])

        q_vec = emb_vecs_loaded[[0, ]]
        print(q_vec.shape)

        with open(args.index_save_path, "rb") as f:
            index_loaded = faiss.deserialize_index(pickle.load(f))
            search_sample(index_loaded, q_vec)


if __name__ == '__main__':
    # parameters setting
    parser = argparse.ArgumentParser(description='Generate data for testing')
    parser.add_argument('--row_nums', type=int, default=200000,
                        help='The number of vectors to be generated')
    parser.add_argument('--vec_dim', type=int, default=256,
                        help='The dimension of vector')
    parser.add_argument('--header_len', type=int, default=8,
                        help='The length of the unique item header')

    parser.add_argument('--verbose', action='store_true',
                        help='Print more detailed information')

    parser.add_argument('--emb_path', type=str, default='./emb_vecs.pkl',
                        help='the path to save vector embeddings')
    parser.add_argument('--dict_path', type=str, default='./item_dict.pkl',
                        help='the path to save item_dict')
    parser.add_argument('--index_save_path', type=str, default='./index_FlatL2.pkl',
                        help='the path to save faiss index data')
    parser.add_argument('--parquet_path', type=str, default='./data.parquet/',
                        help='the path to save vector embeddings with spark')

    parser.add_argument('--index_type', type=str, default='FlatL2',
                        help='index_type: FlatL2 or IVFFlatL2')

    args = parser.parse_args()
    generate_data(args)
