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
import time
import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler

from zoo.orca import init_orca_context, stop_orca_context
from zoo.chronos.autots.autotsestimator import AutoTSEstimator
from zoo.chronos.data.repo_dataset import get_public_dataset


def get_tsdata():
    name = 'nyc_taxi'
    path = '~/.chronos/dataset/'
    tsdata_train, tsdata_val, \
        tsdata_test = get_public_dataset(name,
                                         path,
                                         with_split=True,
                                         val_ratio=0.1,
                                         test_ratio=0.1)
    stand = StandardScaler()
    for tsdata in [tsdata_train, tsdata_val, tsdata_test]:
        tsdata.gen_dt_feature(one_hot_features=['HOUR', 'WEEK'])\
              .impute("last")\
              .scale(stand, fit=tsdata is tsdata_train)
    return tsdata_train, tsdata_val, tsdata_test

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=2,
                        help="The number of nodes to be used in the cluster. "
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--cluster_mode', type=str, default='local',
                        help="The mode for the Spark cluster.")
    parser.add_argument('--cores', type=int, default=4,
                        help="The number of cpu cores you want to use on each node."
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--memory', type=str, default="10g",
                        help="The memory you want to use on each node."
                        "You can change it depending on your own cluster setting.")

    parser.add_argument("--epochs", type=int, default=2,
                        help="Max number of epochs to train in each trial.")
    parser.add_argument("--n_sampling", type=int, default=1,
                        help="Number of times to sample from the search_space.")
    args = parser.parse_args()

    # init_orca_context
    num_nodes = 1 if args.cluster_mode == "local" else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      memory=args.memory, num_nodes=num_nodes, init_ray_on_spark=True)
    tsdata_train, tsdata_val, tsdata_test = get_tsdata()

    autoest = AutoTSEstimator(model='lstm',
                              search_space="normal",
                              past_seq_len=40,
                              future_seq_len=1,
                              cpus_per_trial=2,
                              metric='mse',
                              name='auto_lstm')

    tsppl = autoest.fit(data=tsdata_train,
                        validation_data=tsdata_val,
                        epochs=args.epochs,
                        n_sampling=args.n_sampling)

    mse, smape = tsppl.evaluate(tsdata_test,
                                metrics=['mse', 'smape'])
    print(f'evaluate mse is: {np.mean(mse)}')
    print(f'evaluate smape is: {np.mean(smape)}')

    mse, smape = tsppl.evaluate_with_onnx(tsdata_test, metrics=['mse', 'smape'])
    print(f'evaluate_onnx mse is: {np.mean(mse)}')
    print(f'evaluate_onnx smape is: {np.mean(smape)}')

    start_time = time.time()
    tsppl.predict(tsdata_test)
    print(f'inference time is: {(time.time()-start_time):.3f}s')

    start_time = time.time()
    tsppl.predict_with_onnx(tsdata_test)
    print(f'inference(onnx) time is: {(time.time()-start_time):.3f}s')

    # stop orca
    stop_orca_context()
