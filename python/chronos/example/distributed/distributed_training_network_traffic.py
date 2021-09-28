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
import argparse
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from zoo.orca.automl.metrics import Evaluator
from zoo.orca import init_orca_context, stop_orca_context
from zoo.chronos.forecaster.seq2seq_forecaster import Seq2SeqForecaster
from zoo.chronos.data.repo_dataset import get_public_dataset


def get_tsdata():
    name = 'network_traffic'
    path = '~/.chronos/dataset'
    tsdata_train, _,\
        tsdata_test = get_public_dataset(name, path,
                                         redownload=False,
                                         with_split=True,
                                         test_ratio=0.1)
    minmax = MinMaxScaler()
    for tsdata in [tsdata_train, tsdata_test]:
        tsdata.gen_dt_feature(one_hot_features=['HOUR', 'WEEK'])\
              .impute("last")\
              .scale(minmax, fit=tsdata is tsdata_train)\
              .roll(lookback=100, horizon=10)
    return tsdata_train, tsdata_test


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cores', type=int, default=4,
                        help="The number of cpu cores you want to use on each node."
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--memory', type=str, default="32g",
                        help="The memory you want to use on each node."
                        "You can change it depending on your own cluster setting.")
    parser.add_argument('--cluster_mode', type=str, default='local',
                        help="The mode for the Spark cluster.")
    parser.add_argument('--num_workers', type=int, default=1,
                        help="The number of nodes to be used in the cluster"
                        "You can change it depending on your own cluster setting.")

    parser.add_argument('--epochs', type=int, default=2,
                        help="Max number of epochs to train in each trial.")
    parser.add_argument('--workers_per_node', type=int, default=1,
                        help="the number of worker you want to use."
                        "The value defaults to 1. The param is only effective"
                        "when distributed is set to True.")

    args = parser.parse_args()
    num_nodes = 1 if args.cluster_mode == 'local' else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      memory=args.memory, num_nodes=num_nodes)

    tsdata_train, tsdata_test = get_tsdata()
    x_train, y_train = tsdata_train.to_numpy()
    x_test, y_test = tsdata_test.to_numpy()

    forecaster = Seq2SeqForecaster(past_seq_len=100,
                                   future_seq_len=10,
                                   input_feature_num=x_train.shape[-1],
                                   output_feature_num=2,
                                   metrics=['mse'],
                                   distributed=True,
                                   workers_per_node=args.workers_per_node,
                                   seed=0)

    forecaster.fit((x_train, y_train), epochs=args.epochs,
                   batch_size=512//(1 if not forecaster.distributed else args.workers_per_node))

    yhat = forecaster.predict(x_test)
    unscale_yhat = tsdata_test.unscale_numpy(yhat)
    unscale_y_test = tsdata_test.unscale_numpy(y_test)
    rmse, smape = [Evaluator.evaluate(m, y_true=unscale_y_test,
                                      y_pred=unscale_yhat,
                                      multioutput='raw_values') for m in ['rmse', 'smape']]
    print(f'rmse is: {np.mean(rmse)}')
    print(f'smape is: {np.mean(smape):.4f}')

    stop_orca_context()
