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

import pytest

from bigdl.orca.test_zoo_utils import ZooTestCase
from bigdl.chronos.autots.deprecated.feature.utils import save, restore
from bigdl.chronos.autots.deprecated.feature.time_sequence import *
from numpy.testing import assert_array_almost_equal
import json
import tempfile
import shutil

from bigdl.chronos.autots.deprecated.preprocessing.utils import train_val_test_split


class TestTimeSequenceFeature(ZooTestCase):
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        pass

    def test_get_feature_list(self):
        dates = pd.date_range('1/1/2019', periods=8)
        data = np.random.randn(8, 3)
        df = pd.DataFrame({"datetime": dates, "values": data[:, 0],
                           "A": data[:, 1], "B": data[:, 2]})
        feat = TimeSequenceFeatureTransformer(dt_col="datetime",
                                              target_col="values",
                                              extra_features_col=["A", "B"],
                                              drop_missing=True)
        feature_list = feat.get_feature_list()
        assert set(feature_list) == {'IS_AWAKE(datetime)',
                                     'IS_BUSY_HOURS(datetime)',
                                     'HOUR(datetime)',
                                     'DAY(datetime)',
                                     'IS_WEEKEND(datetime)',
                                     'WEEKDAY(datetime)',
                                     'MONTH(datetime)',
                                     'DAYOFYEAR(datetime)',
                                     'WEEKOFYEAR(datetime)',
                                     'MINUTE(datetime)',
                                     'A',
                                     'B'}

        feat = TimeSequenceFeatureTransformer(dt_col="datetime",
                                              target_col="values",
                                              extra_features_col=["A", "B"],
                                              drop_missing=True,
                                              time_features=False)
        feature_list = feat.get_feature_list()
        assert set(feature_list) == {'A', 'B'}

    def test_fit_transform(self):
        sample_num = 8
        past_seq_len = 2
        dates = pd.date_range('1/1/2019', periods=sample_num)
        data = np.random.randn(sample_num, 3)
        df = pd.DataFrame({"datetime": dates, "values": data[:, 0],
                           "A": data[:, 1], "B": data[:, 2]})
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)',
                                                   'A']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values", drop_missing=True)
        x, y = feat.fit_transform(df, **config)
        assert x.shape == (sample_num - past_seq_len,
                           past_seq_len,
                           len(json.loads(config["selected_features"])) + 1)
        assert y.shape == (sample_num - past_seq_len, 1)
        assert np.mean(np.concatenate((x[0, :, 0], y[:, 0]), axis=None)) < 1e-5

    def test_fit_transform_df_list(self):
        sample_num = 8
        past_seq_len = 2
        dates = pd.date_range('1/1/2019', periods=sample_num)
        data = np.random.randn(sample_num, 3)
        df = pd.DataFrame({"datetime": dates, "values": data[:, 0],
                           "A": data[:, 1], "B": data[:, 2]})
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)',
                                                   'A']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values", drop_missing=True)

        df_list = [df] * 3
        x, y = feat.fit_transform(df_list, **config)
        single_result_len = sample_num - past_seq_len
        assert x.shape == (single_result_len * 3,
                           past_seq_len,
                           len(json.loads(config["selected_features"])) + 1)
        assert y.shape == (single_result_len * 3, 1)
        assert_array_almost_equal(x[:single_result_len],
                                  x[single_result_len: 2 * single_result_len], decimal=2)
        assert_array_almost_equal(x[:single_result_len], x[2 * single_result_len:], decimal=2)
        assert_array_almost_equal(y[:single_result_len],
                                  y[single_result_len: 2 * single_result_len],
                                  decimal=2)
        assert_array_almost_equal(y[:single_result_len], y[2 * single_result_len:], decimal=2)

        assert np.mean(np.concatenate((x[0, :, 0], y[:single_result_len, 0]), axis=None)) < 1e-5

    def test_fit_transform_input_datetime(self):
        # if the type of input datetime is not datetime64, raise an error
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        df = pd.DataFrame({"datetime": dates.strftime('%m/%d/%Y'), "values": values})
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": 2}
        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values", drop_missing=True)

        with pytest.raises(ValueError) as excinfo:
            feat.fit_transform(df, **config)
        assert 'np.datetime64' in str(excinfo.value)

        # if there is NaT in datetime, raise an error
        df.loc[1, "datetime"] = None
        with pytest.raises(ValueError, match=r".* datetime .*"):
            feat.fit_transform(df, **config)

    def test_input_data_len(self):
        sample_num = 100
        past_seq_len = 20
        dates = pd.date_range('1/1/2019', periods=sample_num)
        values = np.random.randn(sample_num)
        df = pd.DataFrame({"datetime": dates, "values": values})
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": past_seq_len}
        train_df, val_df, test_df = train_val_test_split(df,
                                                         val_ratio=0.1,
                                                         test_ratio=0.1,
                                                         look_back=10)

        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values", drop_missing=True)
        with pytest.raises(ValueError, match=r".*past sequence length.*"):
            feat.fit_transform(train_df[:20], **config)

        feat.fit_transform(train_df, **config)
        with pytest.raises(ValueError, match=r".*past sequence length.*"):
            feat.transform(val_df, is_train=True)

        with pytest.raises(ValueError, match=r".*past sequence length.*"):
            feat.transform(test_df[:-1], is_train=False)
        out_x, out_y = feat.transform(test_df, is_train=False)
        assert len(out_x) == 1
        assert out_y is None

    def test_fit_transform_input_data(self):
        # if there is NaN in data other than datetime, drop the training sample.
        num_samples = 8
        dates = pd.date_range('1/1/2019', periods=num_samples)
        values = np.random.randn(num_samples)
        df = pd.DataFrame({"datetime": dates, "values": values})
        df.loc[2, "values"] = None
        past_seq_len = 2

        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values", drop_missing=True)

        x, y = feat.fit_transform(df, **config)
        # mask_x = [1, 0, 0, 1, 1, 1]
        # mask_y = [0, 1, 1, 1, 1, 1]
        # mask   = [0, 0, 0, 1, 1, 1]
        assert x.shape == (3, past_seq_len, len(json.loads(config["selected_features"])) + 1)
        assert y.shape == (3, 1)

    def test_transform_train_true(self):
        num_samples = 16
        dates = pd.date_range('1/1/2019', periods=num_samples)
        values = np.random.randn(num_samples, 2)
        df = pd.DataFrame({"datetime": dates, "values": values[:, 0], "feature_1": values[:, 1]})
        train_sample_num = 10
        train_df = df[:train_sample_num]
        val_df = df[train_sample_num:]
        past_seq_len = 2

        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)',
                                                   "feature_1"]),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values",
                                              extra_features_col="feature_1",
                                              drop_missing=True)

        feat.fit_transform(train_df, **config)
        val_x, val_y = feat.transform(val_df, is_train=True)
        assert val_x.shape == (val_df.shape[0] - past_seq_len,
                               past_seq_len,
                               len(json.loads(config["selected_features"])) + 1)
        assert val_y.shape == (val_df.shape[0] - past_seq_len, 1)

    def test_transform_train_true_df_list(self):
        num_samples = 16
        dates = pd.date_range('1/1/2019', periods=num_samples)
        values = np.random.randn(num_samples, 2)
        df = pd.DataFrame({"datetime": dates, "values": values[:, 0], "feature_1": values[:, 1]})
        train_sample_num = 10
        train_df = df[:train_sample_num]
        val_df = df[train_sample_num:]
        past_seq_len = 2

        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)',
                                                   "feature_1"]),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values",
                                              extra_features_col="feature_1",
                                              drop_missing=True)

        train_df_list = [train_df] * 3
        feat.fit_transform(train_df_list, **config)
        val_df_list = [val_df] * 3
        val_x, val_y = feat.transform(val_df_list, is_train=True)
        single_result_len = val_df.shape[0] - past_seq_len
        assert val_x.shape == (single_result_len * 3,
                               past_seq_len,
                               len(json.loads(config["selected_features"])) + 1)
        assert val_y.shape == (single_result_len * 3, 1)

    def test_transform_train_false(self):
        num_samples = 16
        dates = pd.date_range('1/1/2019', periods=num_samples)
        values = np.random.randn(num_samples, 2)
        df = pd.DataFrame({"datetime": dates, "values": values[:, 0], "feature_1": values[:, 1]})
        train_sample_num = 10
        train_df = df[:train_sample_num]
        test_df = df[train_sample_num:]
        past_seq_len = 2

        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)',
                                                   "feature_1"]),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values",
                                              extra_features_col="feature_1",
                                              drop_missing=True)
        feat.fit_transform(train_df, **config)
        test_x, _ = feat.transform(test_df, is_train=False)
        assert test_x.shape == (test_df.shape[0] - past_seq_len + 1,
                                past_seq_len,
                                len(json.loads(config["selected_features"])) + 1)

    def test_transform_train_false_df_list(self):
        num_samples = 16
        dates = pd.date_range('1/1/2019', periods=num_samples)
        values = np.random.randn(num_samples, 2)
        df = pd.DataFrame({"datetime": dates, "values": values[:, 0], "feature_1": values[:, 1]})
        train_sample_num = 10
        train_df = df[:train_sample_num]
        test_df = df[train_sample_num:]
        past_seq_len = 2

        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)',
                                                   "feature_1"]),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=1, dt_col="datetime",
                                              target_col="values",
                                              extra_features_col="feature_1",
                                              drop_missing=True)
        train_df_list = [train_df] * 3
        feat.fit_transform(train_df_list, **config)
        test_df_list = [test_df] * 3
        test_x, _ = feat.transform(test_df_list, is_train=False)
        assert test_x.shape == ((test_df.shape[0] - past_seq_len + 1) * 3,
                                past_seq_len,
                                len(json.loads(config["selected_features"])) + 1)

    def test_save_restore(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        df = pd.DataFrame({"dt": dates, "v": values})

        future_seq_len = 2
        dt_col = "dt"
        target_col = "v"
        drop_missing = True
        feat = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len,
                                              dt_col=dt_col,
                                              target_col=target_col,
                                              drop_missing=drop_missing)

        feature_list = feat.get_feature_list()
        config = {"selected_features": json.dumps(feature_list),
                  "past_seq_len": 2
                  }

        train_x, train_y = feat.fit_transform(df, **config)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature")
        try:
            save(dirname, feature_transformers=feat)
            new_ft = TimeSequenceFeatureTransformer()
            restore(dirname, feature_transformers=new_ft, config=config)

            assert new_ft.future_seq_len == future_seq_len
            assert new_ft.dt_col == dt_col
            assert new_ft.target_col[0] == target_col
            assert new_ft.extra_features_col is None
            assert new_ft.drop_missing == drop_missing

            test_x, _ = new_ft.transform(df[:-future_seq_len], is_train=False)

            assert_array_almost_equal(test_x, train_x, decimal=2)

        finally:
            shutil.rmtree(dirname)

    def test_post_processing_train(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        dt_col = "datetime"
        value_col = "values"
        df = pd.DataFrame({dt_col: dates, value_col: values})

        past_seq_len = 2
        future_seq_len = 1
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len, dt_col="datetime",
                                              target_col="values", drop_missing=True)

        train_x, train_y = feat.fit_transform(df, **config)
        y_unscale, y_unscale_1 = feat.post_processing(df, train_y, is_train=True)
        y_input = df[past_seq_len:][[value_col]].values
        msg = "y_unscale is {}, y_unscale_1 is {}".format(y_unscale, y_unscale_1)
        assert_array_almost_equal(y_unscale, y_unscale_1, decimal=2), msg
        msg = "y_unscale is {}, y_input is {}".format(y_unscale, y_input)
        assert_array_almost_equal(y_unscale, y_input, decimal=2), msg

    def test_post_processing_train_df_list(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        dt_col = "datetime"
        value_col = "values"
        df = pd.DataFrame({dt_col: dates, value_col: values})

        past_seq_len = 2
        future_seq_len = 1
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len, dt_col="datetime",
                                              target_col="values", drop_missing=True)
        df_list = [df] * 3
        train_x, train_y = feat.fit_transform(df_list, **config)
        y_unscale, y_unscale_1 = feat.post_processing(df_list, train_y, is_train=True)
        y_input = df[past_seq_len:][[value_col]].values
        target_y = np.concatenate([y_input] * 3)
        msg = "y_unscale is {}, y_unscale_1 is {}".format(y_unscale, y_unscale_1)
        assert_array_almost_equal(y_unscale, y_unscale_1, decimal=2), msg
        msg = "y_unscale is {}, y_input is {}".format(y_unscale, target_y)
        assert_array_almost_equal(y_unscale, target_y, decimal=2), msg

    def test_post_processing_test_1(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        dt_col = "datetime"
        value_col = "values"
        df = pd.DataFrame({dt_col: dates, value_col: values})

        past_seq_len = 2
        future_seq_len = 1
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len, dt_col="datetime",
                                              target_col="values", drop_missing=True)

        train_x, train_y = feat.fit_transform(df, **config)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature_")
        try:
            save(dirname, feature_transformers=feat)
            new_ft = TimeSequenceFeatureTransformer()
            restore(dirname, feature_transformers=new_ft, config=config)

            test_df = df[:-future_seq_len]
            new_ft.transform(test_df, is_train=False)
            output_value_df = new_ft.post_processing(test_df, train_y, is_train=False)

            # train_y is generated from df[past_seq_len:]
            target_df = df[past_seq_len:].copy().reset_index(drop=True)

            assert output_value_df[dt_col].equals(target_df[dt_col])
            assert_array_almost_equal(output_value_df[value_col].values,
                                      target_df[value_col].values, decimal=2)

        finally:
            shutil.rmtree(dirname)

    def test_post_processing_test_df_list(self):
        dates = pd.date_range('1/1/2019', periods=8)
        values = np.random.randn(8)
        dt_col = "datetime"
        value_col = "values"
        df = pd.DataFrame({dt_col: dates, value_col: values})

        past_seq_len = 2
        future_seq_len = 1
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len, dt_col="datetime",
                                              target_col="values", drop_missing=True)
        df_list = [df] * 3
        train_x, train_y = feat.fit_transform(df_list, **config)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature_")
        try:
            save(dirname, feature_transformers=feat)
            new_ft = TimeSequenceFeatureTransformer()
            restore(dirname, feature_transformers=new_ft, config=config)

            test_df = df[:-future_seq_len]
            test_df_list = [test_df] * 3
            new_ft.transform(test_df_list, is_train=False)
            output_value_df_list = new_ft.post_processing(test_df_list, train_y, is_train=False)

            # train_y is generated from df[past_seq_len:]
            target_df = df[past_seq_len:].copy().reset_index(drop=True)

            assert output_value_df_list[0].equals(output_value_df_list[1])
            assert output_value_df_list[0].equals(output_value_df_list[2])
            assert output_value_df_list[0][dt_col].equals(target_df[dt_col])
            assert_array_almost_equal(output_value_df_list[0][value_col].values,
                                      target_df[value_col].values, decimal=2)

        finally:
            shutil.rmtree(dirname)

    def test_post_processing_test_2(self):
        sample_num = 8
        dates = pd.date_range('1/1/2019', periods=sample_num)
        values = np.random.randn(sample_num)
        dt_col = "datetime"
        value_col = "values"
        df = pd.DataFrame({dt_col: dates, value_col: values})

        past_seq_len = 2
        future_seq_len = 2
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len, dt_col="datetime",
                                              target_col="values", drop_missing=True)

        train_x, train_y = feat.fit_transform(df, **config)

        dirname = tempfile.mkdtemp(prefix="automl_test_feature_")
        try:
            save(dirname, feature_transformers=feat)
            new_ft = TimeSequenceFeatureTransformer()
            restore(dirname, feature_transformers=new_ft, config=config)

            test_df = df[:-future_seq_len]
            new_ft.transform(test_df, is_train=False)
            output_value_df = new_ft.post_processing(test_df, train_y, is_train=False)
            assert output_value_df.shape == (sample_num - past_seq_len - future_seq_len + 1,
                                             future_seq_len + 1)

            columns = ["{}_{}".format(value_col, i) for i in range(future_seq_len)]
            output_value = output_value_df[columns].values
            target_df = df[past_seq_len:].copy().reset_index(drop=True)
            target_value = feat._roll_test(target_df["values"], future_seq_len)

            assert output_value_df[dt_col].equals(target_df[:-future_seq_len + 1][dt_col])
            msg = "output_value is {}, target_value is {}".format(output_value, target_value)
            assert_array_almost_equal(output_value, target_value, decimal=2), msg

        finally:
            shutil.rmtree(dirname)

    def test_future_time_validation(self):
        sample_num = 8
        dates = pd.date_range('1/1/2100', periods=sample_num)
        values = np.random.randn(sample_num)
        dt_col = "datetime"
        value_col = "values"
        df = pd.DataFrame({dt_col: dates, value_col: values})

        past_seq_len = 2
        future_seq_len = 1
        config = {"selected_features": json.dumps(['IS_AWAKE(datetime)',
                                                   'IS_BUSY_HOURS(datetime)',
                                                   'HOUR(datetime)']),
                  "past_seq_len": past_seq_len}
        feat = TimeSequenceFeatureTransformer(future_seq_len=future_seq_len, dt_col="datetime",
                                              target_col="values", drop_missing=True)
        x, y = feat.fit_transform(df, **config)
        assert x.shape == (sample_num - past_seq_len,
                           past_seq_len,
                           len(json.loads(config["selected_features"])) + 1)
        assert y.shape == (sample_num - past_seq_len, 1)


if __name__ == "__main__":
    pytest.main([__file__])
