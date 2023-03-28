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

import numpy as np
import pandas as pd
import tempfile
import os

from bigdl.chronos.utils import LazyImport
torch = LazyImport('torch')
AutoformerForecaster = LazyImport('bigdl.chronos.forecaster.autoformer_forecaster.AutoformerForecaster')
from bigdl.chronos.data import TSDataset
from unittest import TestCase
import pytest
from .. import op_torch, op_automl


def get_ts_df():
    sample_num = np.random.randint(1000, 1500)
    train_df = pd.DataFrame({"datetime": pd.date_range('1/1/2019', periods=sample_num, freq="1s"),
                             "value": np.random.randn(sample_num),
                             "id": np.array(['00']*sample_num),
                             "extra feature": np.random.randn(sample_num)})
    return train_df


def create_data(loader=False, extra_feature=False):
    df = get_ts_df()
    if extra_feature:
        target = ["value"]
        extra = ["extra feature"]
    else:
        target = ["value", "extra feature"]
        extra = []
    tsdata_train, tsdata_val, tsdata_test =\
        TSDataset.from_pandas(df, dt_col="datetime", target_col=target, extra_feature_col=extra,
                              with_split=True, test_ratio=0.1, val_ratio=0.1)
    if loader is True:
        train_loader = tsdata_train.to_torch_data_loader(lookback=24, horizon=5, roll=True,
                                                        time_enc=True, label_len=12)
        val_loader = tsdata_val.to_torch_data_loader(lookback=24, horizon=5, roll=True,
                                                    time_enc=True, label_len=12, shuffle=False)
        test_loader = tsdata_test.to_torch_data_loader(lookback=24, horizon=5, roll=True,
                                                    time_enc=True, label_len=12, shuffle=False,
                                                    is_predict=True)
        return train_loader, val_loader, test_loader
    else:
        train_data = tsdata_train.roll(lookback=24, horizon=5, time_enc=True, label_len=12).to_numpy()
        val_data = tsdata_val.roll(lookback=24, horizon=5, time_enc=True, label_len=12).to_numpy()
        test_data = tsdata_test.roll(lookback=24, horizon=5, time_enc=True, label_len=12,
                                     is_predict=True).to_numpy()
        train_data = tuple(map(lambda x: x.astype(np.float32), train_data))
        val_data = tuple(map(lambda x: x.astype(np.float32), val_data))
        test_data = tuple(map(lambda x: x.astype(np.float32), test_data))
        return train_data, val_data, test_data


def create_tsdataset(val_ratio=0):
    from bigdl.chronos.data import TSDataset
    import pandas as pd
    timeserious = pd.date_range(start='2020-01-01', freq='s', periods=1000)
    df = pd.DataFrame(np.random.rand(1000, 2),
                      columns=['value1', 'value2'],
                      index=timeserious,
                      dtype=np.float32)
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'timeserious'}, inplace=True)
    train, val, test = TSDataset.from_pandas(df=df,
                                           dt_col='timeserious',
                                           target_col=['value1', 'value2'],
                                           with_split=True,
                                           val_ratio=val_ratio)
    if val_ratio == 0:
        for tsdata in [train, test]:
            tsdata.roll(lookback=24,
                        horizon=5,
                        time_enc=True,
                        label_len=12)
        return train, test
    else:
        for tsdata in [train, val, test]:
            tsdata.roll(lookback=24,
                        horizon=5,
                        time_enc=True,
                        label_len=12)
        return train, val, test

@op_torch
class TestChronosModelAutoformerForecaster(TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_autoformer_forecaster_fit_eval_pred_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s')
        forecaster.fit(train_loader, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_loader)
        pred = forecaster.predict(test_loader)
        
    def test_autoformer_forecaster_fit_eval_pred_array(self):
        train_data, val_data, test_data = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s')
        forecaster.fit(train_data, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_data)
        pred = forecaster.predict(test_data)

    def test_autoformer_forecaster_fit_eval_pred_tsdataset(self):
        train, val, test = create_tsdataset(val_ratio=0.1)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s')
        forecaster.fit(train, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val)
        pred = forecaster.predict(test)

    @op_automl
    def test_autoformer_forecaster_tune(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, test_data = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          loss="mse",
                                          metrics=['mae', 'mse', 'mape'],
                                          lr=space.Real(0.001, 0.01, log=True))
        forecaster.tune(train_data, validation_data=val_data, n_trials=2)
        forecaster.fit(train_data, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_data)

    @op_automl
    def test_autoformer_forecaster_fit_without_tune(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, test_data = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          loss="mse",
                                          metrics=['mae', 'mse', 'mape'],
                                          lr=space.Real(0.001, 0.01, log=True))
        with pytest.raises(RuntimeError) as e:
            forecaster.fit(train_data, epochs=3, batch_size=32)
        error_msg = e.value.args[0]
        assert error_msg == "There is no trainer, and you " \
                            "should call .tune() before .fit()"

    @op_automl
    def test_autoformer_forecaster_multi_objective_tune(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, test_data = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          loss="mse",
                                          metrics=['mae', 'mse', 'mape'],
                                          lr=space.Real(0.001, 0.01, log=True))
        forecaster.tune(train_data, validation_data=val_data, 
                        target_metric=['mse', 'latency'],
                        directions=["minimize", "minimize"],
                        direction=None,
                        n_trials=2)
        forecaster.fit(train_data, epochs=3, batch_size=32, use_trial_id=0)
        evaluate = forecaster.evaluate(val_data)

    def test_autoformer_forecaster_seed(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        evaluate_list = []
        for i in range(2):
            forecaster = AutoformerForecaster(past_seq_len=24,
                                            future_seq_len=5,
                                            input_feature_num=2,
                                            output_feature_num=2,
                                            label_len=12,
                                            freq='s',
                                            seed=0)
            forecaster.fit(train_loader, epochs=3, batch_size=32)
            evaluate = forecaster.evaluate(val_loader)
            pred = forecaster.predict(test_loader)
            evaluate_list.append(evaluate)
        assert evaluate_list[0][0]['val_loss'] == evaluate_list[1][0]['val_loss']

    def test_autoformer_forecaster_save_load(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s')
        forecaster.fit(train_loader, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_loader)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "af.ckpt")
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            evaluate2 = forecaster.evaluate(val_loader)
        assert evaluate[0]['val_loss'] == evaluate2[0]['val_loss']

    @op_automl
    def test_autoformer_forecaster_tune_save_load(self):
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, _ = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                        future_seq_len=5,
                                        input_feature_num=2,
                                        output_feature_num=2,
                                        label_len=12,
                                        d_model=space.Categorical(128, 10),
                                        freq='s',
                                        loss="mse",
                                        metrics=['mae', 'mse', 'mape'],
                                        seed=1,
                                        lr=0.01)
        forecaster.tune(train_data, validation_data=val_data, n_trials=2)
        forecaster.fit(train_data, epochs=3, batch_size=32)
        evaluate1 = forecaster.evaluate(val_data)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            ckpt_name = os.path.join(tmp_dir_name, "tune.ckpt")
            forecaster.save(ckpt_name)
            forecaster.load(ckpt_name)
            evaluate2 = forecaster.evaluate(val_data)
        assert evaluate1[0]['val/loss'] == evaluate2[0]['val_loss']

    def test_autoformer_forecaster_even_kernel(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        evaluate_list = []
        forecaster = AutoformerForecaster(past_seq_len=24,
                                            future_seq_len=5,
                                            input_feature_num=2,
                                            output_feature_num=2,
                                            label_len=12,
                                            freq='s',
                                            seed=0,
                                            moving_avg=20) # even
        forecaster.fit(train_loader, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_loader)
        pred = forecaster.predict(test_loader)
        evaluate_list.append(evaluate)

    def test_autoformer_forecaster_diff_input_output_dim(self):
        train_loader, val_loader, test_loader = create_data(loader=True, extra_feature=True)
        evaluate_list = []
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=1,
                                          label_len=12,
                                          freq='s',
                                          seed=0,
                                          moving_avg=20) # even
        forecaster.fit(train_loader, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_loader)
        pred = forecaster.predict(test_loader)
        evaluate_list.append(evaluate)

    def test_autoformer_forecaster_confidence_interval_with_loader(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          seed=0)
        forecaster.fit(train_loader, epochs=3, batch_size=32)
        # only the first time needs val_data
        y_pred, std = forecaster.predict_interval(data=test_loader,
                                                  validation_data=val_loader,
                                                  repetition_times=5)
        assert y_pred.shape == std.shape
        y_pred, std = forecaster.predict_interval(data=test_loader)

    def test_autoformer_forecaster_confidence_interval_with_numpy(self):
        train_data, val_data, test_data = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          seed=0)
        forecaster.fit(train_data, epochs=3, batch_size=32)
        # only the first time needs val_data
        y_pred, std = forecaster.predict_interval(data=test_data,
                                                  validation_data=val_data,
                                                  repetition_times=5)
        assert y_pred.shape == std.shape
        y_pred, std = forecaster.predict_interval(data=test_data)

    def test_autoformer_forecaster_confidence_interval_with_tsdataset(self):
        train, val, test = create_tsdataset(val_ratio=0.1)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          seed=0,
                                          moving_avg=20) # even
        forecaster.fit(train, epochs=3, batch_size=32)
        # only the first time needs val_data
        y_pred, std = forecaster.predict_interval(data=test,
                                                  validation_data=val,
                                                  repetition_times=5)
        assert y_pred.shape == std.shape
        y_pred, std = forecaster.predict_interval(data=test)

    def test_autoformer_forecaster_from_tsdataset(self):
        df = get_ts_df()
        target = ["value", "extra feature"]
        tsdata_train, tsdata_val, tsdata_test =\
            TSDataset.from_pandas(df, dt_col="datetime", target_col=target,
                                with_split=True, test_ratio=0.1, val_ratio=0.1)
        train_loader = tsdata_train.to_torch_data_loader(lookback=24, horizon=5, time_enc=True)
        forecaster = AutoformerForecaster.from_tsdataset(tsdata_train)
        forecaster.fit(train_loader, epochs=1, batch_size=32)

        # conflict param
        with pytest.raises(RuntimeError):
            forecaster = AutoformerForecaster.from_tsdataset(tsdata_train, label_len=30)

        # no data infer
        tsdata_train, tsdata_val, tsdata_test =\
            TSDataset.from_pandas(df, dt_col="datetime", target_col=target,
                                with_split=True, test_ratio=0.1, val_ratio=0.1)
        forecaster = AutoformerForecaster.from_tsdataset(tsdata_train,
                                                         past_seq_len=24,
                                                         future_seq_len=5)
        forecaster.fit(train_loader, epochs=1, batch_size=32)

    def test_autoformer_forecaster_lookback_equals_to_one(self):
        # from tsdataset
        df = get_ts_df()
        target = ["value", "extra feature"]
        tsdata_train, tsdata_val, tsdata_test =\
            TSDataset.from_pandas(df, dt_col="datetime", target_col=target,
                                with_split=True, test_ratio=0.1, val_ratio=0.1)
        with pytest.raises(RuntimeError):
            forecaster = AutoformerForecaster.from_tsdataset(tsdata_train,
                                                             past_seq_len=1,
                                                             future_seq_len=5)

        # normal definition
        with pytest.raises(RuntimeError):
            forecaster = AutoformerForecaster(past_seq_len=1,
                                            future_seq_len=5,
                                            input_feature_num=2,
                                            output_feature_num=2,
                                            label_len=12,
                                            freq='s')

    def test_autoformer_forecaster_fit_val(self):
        train_data, val_data, _ = create_data()
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          seed=0)
        val_loss = forecaster.fit(train_data, val_data, epochs=10)

    def test_autoformer_forecaster_fit_loader_val(self):
        train_loader, val_loader, _ = create_data(loader=True)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          seed=0)
        val_loss = forecaster.fit(train_loader, val_loader, epochs=10)

    def test_autoformer_forecaster_fit_val_best_epoch(self):
        train_loader, val_loader, _ = create_data(loader=True)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          seed=0)
        val_loss = forecaster.fit(train_loader, val_loader, epochs=10,
                                  validation_mode='best_epoch')

    def test_autoformer_forecaster_fit_val_tsdataset(self):
        train, val, test = create_tsdataset(val_ratio=0.1)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s',
                                          seed=0)
        forecaster.fit(train, val, epochs=3, batch_size=32)
        forecaster.predict(test)

    @op_automl
    def test_autoformer_forecaster_tune_multi_processes(self):
        name = "parallel-example-torch"
        storage = "sqlite:///example_autoformer.db"  # take sqlite for test, recommand to use mysql
        if os.path.exists("./example_autoformer.db"):
            os.remove("./example_autoformer.db")
        import bigdl.nano.automl.hpo.space as space
        train_data, val_data, test_data = create_data(loader=False)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          d_model=space.Categorical(32, 64, 128, 256, 512, 1024),
                                          freq='s',
                                          loss="mse",
                                          metrics=['mae', 'mse', 'mape'],
                                          lr=0.001)
        from bigdl.nano.automl.hpo.backend import SamplerType
        forecaster.tune(train_data, validation_data=val_data, n_trials=5,
                        study_name=name, sampler=SamplerType.Grid,
                        storage=storage, n_parallels=2)
        forecaster.fit(train_data, epochs=3, batch_size=32)
        evaluate = forecaster.evaluate(val_data)
        os.remove("./example_autoformer.db")
        # test remove
        forecaster.tune(train_data, validation_data=val_data, n_trials=2,
                        study_name=name, sampler=SamplerType.Grid,
                        storage=storage, n_parallels=2)

    def test_autoformer_forecaster_ctx_manager(self):
        train_loader, val_loader, test_loader = create_data(loader=True)
        forecaster = AutoformerForecaster(past_seq_len=24,
                                          future_seq_len=5,
                                          input_feature_num=2,
                                          output_feature_num=2,
                                          label_len=12,
                                          freq='s')
        forecaster.fit(train_loader, epochs=1, batch_size=32)

        original_thread = torch.get_num_threads()
        assert forecaster.thread_num == original_thread

        num = max(1, original_thread//2)
        with forecaster.get_context(thread_num=num):
            assert forecaster.context_enabled == True
            current_thread = torch.get_num_threads()
            assert current_thread == num
            pred = forecaster.predict(test_loader)
