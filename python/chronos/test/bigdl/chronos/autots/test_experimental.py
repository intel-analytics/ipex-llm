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

from unittest import TestCase
import pytest

import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from zoo.chronos.autots.experimental import AutoTSTrainer
from zoo.chronos.data.tsdataset import TSDataset
from zoo.orca.automl import hp
import pandas as pd


def get_ts_df():
    sample_num = np.random.randint(100, 200)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                             "value 1": np.random.randn(sample_num),
                             "value 2": np.random.randn(sample_num),
                             "id": np.array(['00'] * sample_num),
                             "extra feature 1": np.random.randn(sample_num),
                             "extra feature 2": np.random.randn(sample_num)})
    return train_df


def get_tsdataset():
    df = get_ts_df()
    return TSDataset.from_pandas(df,
                                 dt_col="datetime",
                                 target_col=["value 1", "value 2"],
                                 extra_feature_col=["extra feature 1", "extra feature 2"],
                                 id_col="id")


def get_data_creator():
    def data_creator(config):
        tsdata = get_tsdataset()
        x, y = tsdata.roll(lookback=7, horizon=1).to_numpy()
        return DataLoader(TensorDataset(torch.from_numpy(x).float(),
                                        torch.from_numpy(y).float()),
                          batch_size=config["batch_size"],
                          shuffle=True)
    return data_creator


class TestAutoTrainer(TestCase):
    def setUp(self) -> None:
        from zoo.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from zoo.orca import stop_orca_context
        stop_orca_context()

    def test_fit_lstm_feature(self):
        input_feature_dim = 11  # This param will not be used
        output_feature_dim = 2  # 2 targets are generated in get_tsdataset

        search_space = {
            'hidden_dim': hp.grid_search([32, 64]),
            'layer_num': hp.randint(1, 3),
            'lr': hp.choice([0.001, 0.003, 0.01]),
            'dropout': hp.uniform(0.1, 0.2)
        }
        auto_trainer = AutoTSTrainer(model='lstm',
                                     search_space=search_space,
                                     past_seq_len=hp.randint(4, 6),
                                     future_seq_len=1,
                                     input_feature_num=input_feature_dim,
                                     output_target_num=output_feature_dim,
                                     selected_features="auto",
                                     metric="mse",
                                     loss=torch.nn.MSELoss(),
                                     logs_dir="/tmp/auto_trainer",
                                     cpus_per_trial=2,
                                     name="auto_trainer")
        auto_trainer.fit(data=get_tsdataset().gen_dt_feature(),
                         epochs=1,
                         batch_size=hp.choice([32, 64]),
                         validation_data=get_tsdataset().gen_dt_feature(),
                         n_sampling=1
                         )
        config = auto_trainer.get_best_config()
        assert 4 <= config["past_seq_len"] <= 6

    def test_fit_tcn_feature(self):
        input_feature_dim = 11  # This param will not be used
        output_feature_dim = 2  # 2 targets are generated in get_tsdataset

        tsdata_train = get_tsdataset().gen_dt_feature()
        tsdata_valid = get_tsdataset().gen_dt_feature()
        tsdata_test = get_tsdataset().gen_dt_feature()

        search_space = {
            'hidden_units': hp.grid_search([32, 64]),
            'levels': hp.randint(4, 6),
            'kernel_size': hp.randint(3, 5),
            'dropout': hp.uniform(0.1, 0.2),
            'lr': hp.loguniform(0.001, 0.01)
        }
        auto_trainer = AutoTSTrainer(model='tcn',
                                     search_space=search_space,
                                     past_seq_len=hp.randint(4, 6),
                                     future_seq_len=1,
                                     input_feature_num=input_feature_dim,
                                     output_target_num=output_feature_dim,
                                     selected_features="auto",
                                     metric="mse",
                                     optimizer="Adam",
                                     loss=torch.nn.MSELoss(),
                                     logs_dir="/tmp/auto_trainer",
                                     cpus_per_trial=2,
                                     name="auto_trainer")
        auto_trainer.fit(data=tsdata_train,
                         epochs=1,
                         batch_size=hp.choice([32, 64]),
                         validation_data=tsdata_valid,
                         n_sampling=1
                         )
        best_config = auto_trainer.get_best_config()
        best_model = auto_trainer.get_best_model()
        assert 4 <= best_config["past_seq_len"] <= 6

        # really difficult to use the model currently...
        tsdata_test.roll(lookback=best_config["past_seq_len"],
                         horizon=1,
                         feature_col=best_config["selected_features"])
        x_test, y_test = tsdata_test.to_numpy()
        y_pred = best_model.predict(x_test)
        best_model.save("best.ckpt")
        from zoo.automl.model.base_pytorch_model import PytorchModelBuilder
        restore_model = PytorchModelBuilder(model_creator=best_model.model_creator,
                                            optimizer_creator="Adam",
                                            loss_creator=torch.nn.MSELoss()).build(best_config)
        restore_model.restore("best.ckpt")
        y_pred_restore = restore_model.predict(x_test)
        np.testing.assert_almost_equal(y_pred, y_pred_restore)

    def test_fit_lstm_data_creator(self):
        input_feature_dim = 4
        output_feature_dim = 2  # 2 targets are generated in get_tsdataset

        search_space = {
            'hidden_dim': hp.grid_search([32, 64]),
            'layer_num': hp.randint(1, 3),
            'lr': hp.choice([0.001, 0.003, 0.01]),
            'dropout': hp.uniform(0.1, 0.2)
        }
        auto_trainer = AutoTSTrainer(model='lstm',
                                     search_space=search_space,
                                     past_seq_len=7,
                                     future_seq_len=1,
                                     input_feature_num=input_feature_dim,
                                     output_target_num=output_feature_dim,
                                     selected_features="auto",
                                     metric="mse",
                                     loss=torch.nn.MSELoss(),
                                     logs_dir="/tmp/auto_trainer",
                                     cpus_per_trial=2,
                                     name="auto_trainer")
        auto_trainer.fit(data=get_data_creator(),
                         epochs=1,
                         batch_size=hp.choice([32, 64]),
                         validation_data=get_data_creator(),
                         n_sampling=1
                         )
        config = auto_trainer.get_best_config()
        assert config["past_seq_len"] == 7


if __name__ == "__main__":
    pytest.main([__file__])
