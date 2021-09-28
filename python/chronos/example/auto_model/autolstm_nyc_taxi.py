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
import torch
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler

from zoo.chronos.data.repo_dataset import get_public_dataset
from zoo.chronos.autots.model.auto_lstm import AutoLSTM
from zoo.orca.automl import hp
from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.automl.metrics import Evaluator


def get_tsdata():
    name = 'nyc_taxi'
    path = '~/.chronos/dataset/'
    tsdata_train, tsdata_valid, \
        tsdata_test = get_public_dataset(name, path,
                                         redownload=False,
                                         with_split=True,
                                         val_ratio=0.1,
                                         test_ratio=0.1)
    stand_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_valid, tsdata_test]:
        tsdata.impute(mode="linear")\
              .scale(stand_scaler, fit=(tsdata is tsdata_train))
    return tsdata_train, tsdata_valid, tsdata_test


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

    parser.add_argument('--epochs', type=int, default=1,
                        help="Max number of epochs to train in each trial.")
    args = parser.parse_args()

    num_nodes = 1 if args.cluster_mode == "local" else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      memory=args.memory, num_nodes=num_nodes, init_ray_on_spark=True)

    tsdata_train, tsdata_valid, tsdata_test = get_tsdata()

    auto_lstm = AutoLSTM(input_feature_num=1,
                         output_target_num=1,
                         past_seq_len=20,
                         hidden_dim=hp.grid_search([32, 64]),
                         layer_num=hp.randint(1, 3),
                         lr=hp.choice([0.01, 0.03, 0.1]),
                         dropout=hp.uniform(0.1, 0.2),
                         optimizer='Adam',
                         loss=torch.nn.MSELoss(),
                         metric="mse")

    x_train, y_train = tsdata_train.roll(lookback=20, horizon=1).to_numpy()
    x_val, y_val = tsdata_test.roll(lookback=20, horizon=1).to_numpy()
    x_test, y_test = tsdata_test.roll(lookback=20, horizon=1).to_numpy()

    auto_lstm.fit(data=(x_train, y_train),
                  epochs=args.epochs,
                  validation_data=(x_val, y_val))

    yhat = auto_lstm.predict(x_test)
    unscale_y_test = tsdata_test.unscale_numpy(y_test)
    unscale_yhat = tsdata_test.unscale_numpy(yhat)

    rmse, smape = [Evaluator.evaluate(m, y_true=unscale_y_test,
                                      y_pred=unscale_yhat) for m in ['rmse', 'smape']]
    print(f'rmse is {np.mean(rmse)}')
    print(f'sampe is {np.mean(smape)}')

    stop_orca_context()
