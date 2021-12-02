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

from unittest import TestCase
import pytest

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from bigdl.chronos.autots import AutoTSEstimator, TSPipeline
from bigdl.chronos.data import TSDataset
from bigdl.orca.automl import hp
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


class CustomizedNet(nn.Module):
    def __init__(self,
                 dropout,
                 input_size,
                 input_feature_num,
                 hidden_dim,
                 output_size):
        '''
        Simply use linear layers for multi-variate single-step forecasting.
        '''
        super().__init__()
        self.fc1 = nn.Linear(input_size*input_feature_num, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        # x.shape = (num_sample, input_size, input_feature_num)
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu1(x)
        x = self.fc2(x)
        # x.shape = (num_sample, output_size)
        x = torch.unsqueeze(x, 1)
        # x.shape = (num_sample, 1, output_size)
        return x


def model_creator(config):
    return CustomizedNet(dropout=config["dropout"],
                         input_size=config["past_seq_len"],
                         input_feature_num=config["input_feature_num"],
                         hidden_dim=config["hidden_dim"],
                         output_size=config["output_feature_num"])


class TestAutoTrainer(TestCase):
    def setUp(self) -> None:
        from bigdl.orca import init_orca_context
        init_orca_context(cores=8, init_ray_on_spark=True)

    def tearDown(self) -> None:
        from bigdl.orca import stop_orca_context
        stop_orca_context()

    def test_fit_third_party_feature(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)

        search_space = {
            'hidden_dim': hp.grid_search([32, 64]),
            'dropout': hp.uniform(0.1, 0.2)
        }

        auto_estimator = AutoTSEstimator(model=model_creator,
                                         search_space=search_space,
                                         past_seq_len=hp.randint(4, 6),
                                         future_seq_len=1,
                                         selected_features="auto",
                                         metric="mse",
                                         loss=torch.nn.MSELoss(),
                                         cpus_per_trial=2)

        ts_pipeline = auto_estimator.fit(data=tsdata_train,
                                         epochs=1,
                                         batch_size=hp.choice([32, 64]),
                                         validation_data=tsdata_valid,
                                         n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config["past_seq_len"] <= 6

        assert isinstance(ts_pipeline, TSPipeline)

        # use raw base model to predic and evaluate
        tsdata_valid.roll(lookback=best_config["past_seq_len"],
                          horizon=0,
                          feature_col=best_config["selected_features"])
        x_valid, y_valid = tsdata_valid.to_numpy()
        y_pred_raw = best_model.predict(x_valid)
        y_pred_raw = tsdata_valid.unscale_numpy(y_pred_raw)

        # use tspipeline to predic and evaluate
        eval_result = ts_pipeline.evaluate(tsdata_valid)
        y_pred = ts_pipeline.predict(tsdata_valid)

        # check if they are the same
        np.testing.assert_almost_equal(y_pred, y_pred_raw)

        # save and load
        ts_pipeline.save("/tmp/auto_trainer/autots_tmp_model_3rdparty")
        new_ts_pipeline = TSPipeline.load("/tmp/auto_trainer/autots_tmp_model_3rdparty")

        # check if load ppl is the same as previous
        eval_result_new = new_ts_pipeline.evaluate(tsdata_valid)
        y_pred_new = new_ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(eval_result[0], eval_result_new[0])
        np.testing.assert_almost_equal(y_pred, y_pred_new)

        # use tspipeline to incrementally train
        new_ts_pipeline.fit(tsdata_valid)

    def test_fit_third_party_data_creator(self):
        input_feature_dim = 4
        output_feature_dim = 2  # 2 targets are generated in get_tsdataset

        search_space = {
            'hidden_dim': hp.grid_search([32, 64]),
            'dropout': hp.uniform(0.1, 0.2)
        }

        auto_estimator = AutoTSEstimator(model=model_creator,
                                         search_space=search_space,
                                         past_seq_len=7,
                                         future_seq_len=1,
                                         input_feature_num=input_feature_dim,
                                         output_target_num=output_feature_dim,
                                         selected_features="auto",
                                         metric="mse",
                                         loss=torch.nn.MSELoss(),
                                         cpus_per_trial=2)

        auto_estimator.fit(data=get_data_creator(),
                           epochs=1,
                           batch_size=hp.choice([32, 64]),
                           validation_data=get_data_creator(),
                           n_sampling=1)

        config = auto_estimator.get_best_config()
        assert config["past_seq_len"] == 7

    def test_fit_lstm_feature(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)

        auto_estimator = AutoTSEstimator(model='lstm',
                                         search_space="minimal",
                                         past_seq_len=hp.randint(4, 6),
                                         future_seq_len=1,
                                         selected_features="auto",
                                         metric="mse",
                                         loss=torch.nn.MSELoss(),
                                         logs_dir="/tmp/auto_trainer",
                                         cpus_per_trial=2,
                                         name="auto_trainer")
        ts_pipeline = auto_estimator.fit(data=tsdata_train,
                                         epochs=1,
                                         batch_size=hp.choice([32, 64]),
                                         validation_data=tsdata_valid,
                                         n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config["past_seq_len"] <= 6

        assert isinstance(ts_pipeline, TSPipeline)

        # use raw base model to predic and evaluate
        tsdata_valid.roll(lookback=best_config["past_seq_len"],
                          horizon=0,
                          feature_col=best_config["selected_features"])
        x_valid, y_valid = tsdata_valid.to_numpy()
        y_pred_raw = best_model.predict(x_valid)
        y_pred_raw = tsdata_valid.unscale_numpy(y_pred_raw)

        # use tspipeline to predic and evaluate
        eval_result = ts_pipeline.evaluate(tsdata_valid)
        y_pred = ts_pipeline.predict(tsdata_valid)

        # check if they are the same
        np.testing.assert_almost_equal(y_pred, y_pred_raw)

        # save and load
        ts_pipeline.save("/tmp/auto_trainer/autots_tmp_model_lstm")
        new_ts_pipeline = TSPipeline.load("/tmp/auto_trainer/autots_tmp_model_lstm")

        # check if load ppl is the same as previous
        eval_result_new = new_ts_pipeline.evaluate(tsdata_valid)
        y_pred_new = new_ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(eval_result[0], eval_result_new[0])
        np.testing.assert_almost_equal(y_pred, y_pred_new)

        # check if load ppl is the same as previous with onnx
        try:
            import onnx
            import onnxruntime
            eval_result_new_onnx = new_ts_pipeline.evaluate_with_onnx(tsdata_valid)
            y_pred_new_onnx = new_ts_pipeline.predict_with_onnx(tsdata_valid)
            np.testing.assert_almost_equal(eval_result[0], eval_result_new_onnx[0], decimal=5)
            np.testing.assert_almost_equal(y_pred, y_pred_new_onnx, decimal=5)
        except ImportError:
            pass

        # use tspipeline to incrementally train
        new_ts_pipeline.fit(tsdata_valid)

    def test_fit_tcn_feature(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)

        auto_estimator = AutoTSEstimator(model='tcn',
                                         search_space="minimal",
                                         past_seq_len=hp.randint(4, 6),
                                         future_seq_len=1,
                                         selected_features="auto",
                                         metric="mse",
                                         optimizer="Adam",
                                         loss=torch.nn.MSELoss(),
                                         logs_dir="/tmp/auto_trainer",
                                         cpus_per_trial=2,
                                         name="auto_trainer")
        ts_pipeline = auto_estimator.fit(data=tsdata_train,
                                         epochs=1,
                                         batch_size=hp.choice([32, 64]),
                                         validation_data=tsdata_valid,
                                         n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config["past_seq_len"] <= 6

        assert isinstance(ts_pipeline, TSPipeline)

        # use raw base model to predic and evaluate
        tsdata_valid.roll(lookback=best_config["past_seq_len"],
                          horizon=0,
                          feature_col=best_config["selected_features"])
        x_valid, y_valid = tsdata_valid.to_numpy()
        y_pred_raw = best_model.predict(x_valid)
        y_pred_raw = tsdata_valid.unscale_numpy(y_pred_raw)

        # use tspipeline to predic and evaluate
        eval_result = ts_pipeline.evaluate(tsdata_valid)
        y_pred = ts_pipeline.predict(tsdata_valid)

        # check if they are the same
        np.testing.assert_almost_equal(y_pred, y_pred_raw)

        # save and load
        ts_pipeline.save("/tmp/auto_trainer/autots_tmp_model_tcn")
        new_ts_pipeline = TSPipeline.load("/tmp/auto_trainer/autots_tmp_model_tcn")

        # check if load ppl is the same as previous
        eval_result_new = new_ts_pipeline.evaluate(tsdata_valid)
        y_pred_new = new_ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(eval_result[0], eval_result_new[0])
        np.testing.assert_almost_equal(y_pred, y_pred_new)

        # check if load ppl is the same as previous with onnx
        try:
            import onnx
            import onnxruntime
            eval_result_new_onnx = new_ts_pipeline.evaluate_with_onnx(tsdata_valid)
            y_pred_new_onnx = new_ts_pipeline.predict_with_onnx(tsdata_valid)
            np.testing.assert_almost_equal(eval_result[0], eval_result_new_onnx[0], decimal=5)
            np.testing.assert_almost_equal(y_pred, y_pred_new_onnx, decimal=5)
        except ImportError:
            pass

        # use tspipeline to incrementally train
        new_ts_pipeline.fit(tsdata_valid)

    def test_fit_seq2seq_feature(self):
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        tsdata_train = get_tsdataset().gen_dt_feature().scale(scaler, fit=True)
        tsdata_valid = get_tsdataset().gen_dt_feature().scale(scaler, fit=False)

        auto_estimator = AutoTSEstimator(model='seq2seq',
                                         search_space="minimal",
                                         past_seq_len=hp.randint(4, 6),
                                         future_seq_len=1,
                                         selected_features="auto",
                                         metric="mse",
                                         optimizer="Adam",
                                         loss=torch.nn.MSELoss(),
                                         logs_dir="/tmp/auto_trainer",
                                         cpus_per_trial=2,
                                         name="auto_trainer")
        ts_pipeline = auto_estimator.fit(data=tsdata_train,
                                         epochs=1,
                                         batch_size=hp.choice([32, 64]),
                                         validation_data=tsdata_valid,
                                         n_sampling=1)
        best_config = auto_estimator.get_best_config()
        best_model = auto_estimator._get_best_automl_model()
        assert 4 <= best_config["past_seq_len"] <= 6

        assert isinstance(ts_pipeline, TSPipeline)

        # use raw base model to predic and evaluate
        tsdata_valid.roll(lookback=best_config["past_seq_len"],
                          horizon=0,
                          feature_col=best_config["selected_features"])
        x_valid, y_valid = tsdata_valid.to_numpy()
        y_pred_raw = best_model.predict(x_valid)
        y_pred_raw = tsdata_valid.unscale_numpy(y_pred_raw)

        # use tspipeline to predic and evaluate
        eval_result = ts_pipeline.evaluate(tsdata_valid)
        y_pred = ts_pipeline.predict(tsdata_valid)

        # check if they are the same
        np.testing.assert_almost_equal(y_pred, y_pred_raw)

        # save and load
        ts_pipeline.save("/tmp/auto_trainer/autots_tmp_model_seq2seq")
        new_ts_pipeline = TSPipeline.load("/tmp/auto_trainer/autots_tmp_model_seq2seq")

        # check if load ppl is the same as previous
        eval_result_new = new_ts_pipeline.evaluate(tsdata_valid)
        y_pred_new = new_ts_pipeline.predict(tsdata_valid)
        np.testing.assert_almost_equal(eval_result[0], eval_result_new[0])
        np.testing.assert_almost_equal(y_pred, y_pred_new)

        # check if load ppl is the same as previous with onnx
        try:
            import onnx
            import onnxruntime
            eval_result_new_onnx = new_ts_pipeline.evaluate_with_onnx(tsdata_valid)
            y_pred_new_onnx = new_ts_pipeline.predict_with_onnx(tsdata_valid)
            np.testing.assert_almost_equal(eval_result[0], eval_result_new_onnx[0], decimal=5)
            np.testing.assert_almost_equal(y_pred, y_pred_new_onnx, decimal=5)
        except ImportError:
            pass

        # use tspipeline to incrementally train
        new_ts_pipeline.fit(tsdata_valid)

    def test_fit_lstm_data_creator(self):
        input_feature_dim = 4
        output_feature_dim = 2  # 2 targets are generated in get_tsdataset

        search_space = {
            'hidden_dim': hp.grid_search([32, 64]),
            'layer_num': hp.randint(1, 3),
            'lr': hp.choice([0.001, 0.003, 0.01]),
            'dropout': hp.uniform(0.1, 0.2)
        }
        auto_estimator = AutoTSEstimator(model='lstm',
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
        auto_estimator.fit(data=get_data_creator(),
                           epochs=1,
                           batch_size=hp.choice([32, 64]),
                           validation_data=get_data_creator(),
                           n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config["past_seq_len"] == 7

    def test_select_feature(self):
        sample_num = np.random.randint(100, 200)
        df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                           "value": np.random.randn(sample_num),
                           "id": np.array(['00']*sample_num)})
        train_ts, val_ts, _ = TSDataset.from_pandas(df,
                                                    target_col=['value'],
                                                    dt_col='datetime',
                                                    id_col='id',
                                                    with_split=True,
                                                    val_ratio=0.1)

        search_space = {
            'hidden_dim': hp.grid_search([32, 64]),
            'layer_num': hp.randint(1, 3),
            'lr': hp.choice([0.001, 0.003, 0.01]),
            'dropout': hp.uniform(0.1, 0.2)
        }

        input_feature_dim, output_feature_dim = 1, 1
        auto_estimator = AutoTSEstimator(model='lstm',
                                         search_space=search_space,
                                         past_seq_len=6,
                                         future_seq_len=1,
                                         input_feature_num=input_feature_dim,
                                         output_target_num=output_feature_dim,
                                         selected_features="auto",
                                         metric="mse",
                                         loss=torch.nn.MSELoss(),
                                         cpus_per_trial=2,
                                         name="auto_trainer")

        auto_estimator.fit(data=train_ts,
                           epochs=1,
                           batch_size=hp.choice([32, 64]),
                           validation_data=val_ts,
                           n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config['past_seq_len'] == 6

    def test_future_list_input(self):
        sample_num = np.random.randint(100, 200)
        df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                           "value": np.random.randn(sample_num),
                           "id": np.array(['00']*sample_num)})
        train_ts, val_ts, _ = TSDataset.from_pandas(df,
                                                    target_col=['value'],
                                                    dt_col='datetime',
                                                    id_col='id',
                                                    with_split=True,
                                                    val_ratio=0.1)

        input_feature_dim, output_feature_dim = 1, 1
        auto_estimator = AutoTSEstimator(model='seq2seq',
                                         search_space="minimal",
                                         past_seq_len=6,
                                         future_seq_len=[1, 3],
                                         input_feature_num=input_feature_dim,
                                         output_target_num=output_feature_dim,
                                         selected_features="auto",
                                         metric="mse",
                                         loss=torch.nn.MSELoss(),
                                         cpus_per_trial=2,
                                         name="auto_trainer")

        auto_estimator.fit(data=train_ts,
                           epochs=1,
                           batch_size=hp.choice([32, 64]),
                           validation_data=val_ts,
                           n_sampling=1)
        config = auto_estimator.get_best_config()
        assert config['future_seq_len'] == 2
        assert auto_estimator._future_seq_len == [1, 3]

    def test_autogener_best_lookback(self):
        sample_num = 100
        df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num),
                           "value": np.arange(sample_num),
                           "id": np.array(['00'] * sample_num)})

        train_ts = TSDataset.from_pandas(df,
                                         target_col=['value'],
                                         dt_col='datetime',
                                         id_col='id',
                                         with_split=False)
        train_ts.roll(horizon=1)

        input_feature_dim, output_feature_dim = 1, 1
        auto_estimator = AutoTSEstimator(model='lstm',
                                         search_space="minimal",
                                         input_feature_num=input_feature_dim,
                                         output_target_num=output_feature_dim)
        
        auto_estimator.fit(data=train_ts,
                           epochs=1,
                           batch_size=hp.choice([16, 32]),
                           validation_data=train_ts)
        config = auto_estimator.get_best_config()
        assert config['past_seq_len'] == 33

if __name__ == "__main__":
    pytest.main([__file__])
