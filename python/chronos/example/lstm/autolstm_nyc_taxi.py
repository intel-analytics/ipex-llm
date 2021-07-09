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
import os

import torch
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import StandardScaler

from torch.utils.data import TensorDataset, DataLoader
from zoo.chronos.data.tsdataset import TSDataset
from zoo.chronos.autots.model.auto_lstm import AutoLSTM
from zoo.orca.automl import hp
from zoo.orca import init_orca_context, stop_orca_context
from zoo.automl.common.metrics import Evaluator


def get_data(path):
    df = pd.read_csv(path, engine='python')
    return df


def get_nyc_taxi_tsdataset(path):
    df = get_data(path)
    tsdata_train, tsdata_valid, tsdata_test = TSDataset.from_pandas(df,
                                                                    dt_col="timestamp",
                                                                    target_col=[
                                                                        "value"],
                                                                    with_split=True,
                                                                    val_ratio=0.1,
                                                                    test_ratio=0.1)
    stand_scaler = StandardScaler()
    for tsdata in [tsdata_train, tsdata_valid, tsdata_test]:
        tsdata.impute(mode="linear")\
            .scale(stand_scaler, fit=(tsdata is tsdata_train))
    return tsdata_train, tsdata_valid, tsdata_test


def get_data_creator(data):
    def data_creator(config):
        x, y = data.roll(lookback=14, horizon=1).to_numpy()
        return DataLoader(TensorDataset(torch.from_numpy(x).float(),
                                        torch.from_numpy(y).float()),
                          batch_size=config["batch_size"],
                          shuffle=True)
    return data_creator


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

    url = 'https://raw.githubusercontent.com/numenta/NAB/v1.0/data/realKnownCause/nyc_taxi.csv'

    parser.add_argument('--epoch', type=int, default=1,
                        help="Max number of epochs to train in each trial.")
    parser.add_argument('--cpus_per_trial', type=int, default=2,
                        help="Int. Number of cpus for each trial")
    parser.add_argument('--n_sampling', type=int, default=1,
                        help="Number of times to sample from the search_space.")
    parser.add_argument('--datadir', default=url,
                        help="wget https://raw.githubusercontent.com/numenta/NAB/"
                        "v1.0/data/realKnownCause/nyc_taxi.csv")

    args = parser.parse_args()

    num_nodes = 1 if args.cluster_mode == "local" else args.num_workers
    init_orca_context(cluster_mode=args.cluster_mode, cores=args.cores,
                      memory=args.memory, num_nodes=num_nodes, init_ray_on_spark=True)

    tsdata_train, tsdata_valid, tsdata_test = get_nyc_taxi_tsdataset(
        args.datadir)

    auto_lstm = AutoLSTM(input_feature_num=1,
                         output_target_num=1,
                         past_seq_len=14,
                         optimizer='Adam',
                         loss=torch.nn.MSELoss(),
                         metric="mse",
                         hidden_dim=hp.grid_search([32, 64]),
                         layer_num=hp.randint(1, 3),
                         lr=hp.choice([0.001, 0.003, 0.01]),
                         dropout=hp.uniform(0.1, 0.2),
                         logs_dir="/tmp/auto_lstm",
                         cpus_per_trial=args.cpus_per_trial,
                         name="auto_lstm")
    auto_lstm.fit(data=get_data_creator(tsdata_train),
                  epochs=args.epoch,
                  batch_size=hp.choice([32, 64]),
                  validation_data=get_data_creator(tsdata_valid),
                  n_sampling=args.n_sampling,
                  )
    best_model = auto_lstm.get_best_model()
    best_config = auto_lstm.get_best_config()

    x, y = tsdata_test\
        .roll(lookback=best_config["past_seq_len"],
              horizon=best_config["future_seq_len"])\
        .to_numpy()
    yhat = best_model(torch.from_numpy(x).float()).detach().numpy()

    y_unscale = tsdata_test.unscale_numpy(y)
    yhat_unscale = tsdata_test.unscale_numpy(np.expand_dims(yhat, axis=1))

    result = [Evaluator.evaluate(m, y_true=y_unscale, y_pred=yhat_unscale,
                                 multioutput="uniform_average") for m in ['rmse', 'smape']]
    print(f'rmse is {result[0]}, sampe is {result[1]}')
    print(f'The hyperparameters of the model are {best_config}')
    stop_orca_context()
