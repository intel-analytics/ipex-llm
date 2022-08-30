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
import argparse
import os
from argparse import ArgumentParser

from bigdl.friesian.feature import FeatureTable
from bigdl.orca import init_orca_context, stop_orca_context


def transform(x):
    if x == '上海':
        return 0.0
    elif isinstance(x, float):
        return float(x)
    else:
        return float(eval(x))


def transform_cat_2(x):
    return '-'.join(sorted(x.split('/')))


def read_and_split(data_input_path, sparse_int_features, sparse_string_features, dense_features):
    header_names = ['user_id', 'article_id', 'expo_time', 'net_status', 'flush_nums',
                    'exop_position', 'click', 'duration', 'device', 'os', 'province', 'city',
                    'age', 'gender', 'ctime', 'img_num', 'cat_1', 'cat_2'
                    ]
    if data_input_path.split('.')[-1] == 'csv':
        data_pd = FeatureTable.read_csv(data_input_path, header=False, names=header_names)
    else:
        data_pd = FeatureTable.read_parquet(data_input_path)
    data_pd = data_pd.cast(sparse_int_features, 'string')
    data_pd = data_pd.cast(dense_features, 'string')

    # fill absence data
    for feature in (sparse_int_features + sparse_string_features):
        data_pd = data_pd.fillna("", feature)
    for dense_feature in dense_features:
        data_pd = data_pd.fillna('0.0', dense_feature)
    print(data_pd.df.dtypes)

    process_img_num = lambda x: transform(x)
    process_cat_2 = lambda x: transform_cat_2(x)
    data_pd = data_pd.apply("img_num", "img_num", process_img_num, "float")
    data_pd = data_pd.apply("cat_2", "cat_2", process_cat_2, "string")

    train_tbl = FeatureTable(data_pd.df[data_pd.df['expo_time'] < '2021-07-06'])
    valid_tbl = FeatureTable(data_pd.df[data_pd.df['expo_time'] >= '2021-07-06'])
    print('train_data.shape: ', train_tbl.size())
    print('test_data.shape: ', valid_tbl.size())
    return train_tbl, valid_tbl


def feature_engineering(train_tbl, valid_tbl, model_path, model_path_json, sparse_int_features,
                        sparse_string_features, dense_features):
    import json
    train_tbl, min_max_dict = train_tbl.min_max_scale(dense_features)
    valid_tbl = valid_tbl.transform_min_max_scale(dense_features, min_max_dict)
    cat_cols = sparse_string_features[-1:] + sparse_int_features + sparse_string_features[:-1]
    for feature in cat_cols:
        train_tbl, feature_idx = train_tbl.category_encode(feature)
        valid_tbl = valid_tbl.encode_string(feature, feature_idx)
        valid_tbl = valid_tbl.fillna(0, feature)
        print("The class number of feature: {}/{}".format(feature, feature_idx.size()))
        feature_idx.write_parquet(model_path)
        fea_dict = feature_idx.to_dict()
        with open(model_path_json + "/" + feature + '.json', 'w', encoding='utf-8') as ff:
            ff.write(json.dumps(fea_dict, ensure_ascii=False, indent=2))
    return train_tbl, valid_tbl


def _parse_args():
    parser = ArgumentParser(description="Transform dataset for multi task demo")
    parser.add_argument('--input_path', type=str,
                        default='/path/to/input/dataset',
                        help='The path for input dataset')
    parser.add_argument('--output_path', type=str, default='/path/to/save/processed/dataset',
                        help='The path for output dataset')
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The cluster mode, such as local, yarn, standalone or spark-submit.')
    parser.add_argument('--master', type=str, default=None,
                        help='The master url, only used when cluster mode is standalone.')
    parser.add_argument('--executor_cores', type=int, default=8,
                        help='The executor core number.')
    parser.add_argument('--executor_memory', type=str, default="12g",
                        help='The executor memory.')
    parser.add_argument('--num_executors', type=int, default=4,
                        help='The number of executors.')
    parser.add_argument('--driver_cores', type=int, default=2,
                        help='The driver core number.')
    parser.add_argument('--driver_memory', type=str, default="8g",
                        help='The driver memory.')
    args_ = parser.parse_args()
    return args_


if __name__ == '__main__':
    args = _parse_args()
    if args.cluster_mode == "local":
        sc = init_orca_context("local", cores=args.executor_cores,
                               memory=args.executor_memory)
    elif args.cluster_mode == "standalone":
        sc = init_orca_context("standalone", master=args.master,
                               cores=args.executor_cores, num_nodes=args.num_executors,
                               memory=args.executor_memory,
                               driver_cores=args.driver_cores,
                               driver_memory=args.driver_memory)
    elif args.cluster_mode == "yarn":
        sc = init_orca_context("yarn-client", cores=args.executor_cores,
                               num_nodes=args.num_executors, memory=args.executor_memory,
                               driver_cores=args.driver_cores, driver_memory=args.driver_memory)
    elif args.cluster_mode == "spark-submit":
        sc = init_orca_context("spark-submit")
    else:
        argparse.ArgumentError(False,
                               "cluster_mode should be one of 'local', 'yarn', 'standalone' and"
                               " 'spark-submit', but got " + args.cluster_mode)

    sparse_int_features_ = [
        'user_id', 'article_id',
        'net_status', 'flush_nums',
        'exop_position',
    ]
    sparse_string_features_ = [
        'device', 'os', 'province',
        'city', 'age',
        'gender', 'cat_1', 'cat_2'
    ]
    dense_features_ = ['img_num']
    model_path_ = os.path.join(args.output_path, 'feature_maps')
    model_path_json_ = os.path.join(args.output_path, 'feature_maps_json')
    os.makedirs(model_path_, exist_ok=True)
    os.makedirs(model_path_json_, exist_ok=True)
    # read, reformat and split data
    df_train, df_test = read_and_split(args.input_path, sparse_int_features_,
                                       sparse_string_features_, dense_features_)
    train_tbl_, valid_tbl_ = feature_engineering(df_train, df_test,
                                                 model_path_, model_path_json_,
                                                 sparse_int_features_,
                                                 sparse_string_features_, dense_features_)
    print(train_tbl_.size())
    print(valid_tbl_.size())
    train_tbl_.write_parquet(os.path.join(args.output_path, 'train_processed'))
    valid_tbl_.write_parquet(os.path.join(args.output_path, 'test_processed'))
    stop_orca_context()
