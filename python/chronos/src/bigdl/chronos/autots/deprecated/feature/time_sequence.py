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

from bigdl.chronos.autots.deprecated.feature.utils import save_config
from bigdl.chronos.autots.deprecated.feature.abstract import BaseFeatureTransformer
from bigdl.chronos.utils import deprecated

import sklearn
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import json
from packaging import version


TIME_FEATURE = ("MINUTE", "DAY", "DAYOFYEAR", "HOUR", "WEEKDAY", "WEEKOFYEAR", "MONTH")
ADDITIONAL_TIME_FEATURE = ("IS_AWAKE", "IS_BUSY_HOURS", "IS_WEEKEND")


class TimeSequenceFeatureTransformer(BaseFeatureTransformer):
    """
    TimeSequence feature engineering
    """

    def __init__(self, future_seq_len=1,
                 dt_col="datetime",
                 target_col=["value"],
                 extra_features_col=None,
                 drop_missing=True,
                 time_features=True):
        """
        Constructor.
        :param future_seq_len: the future sequence length to be predicted
        :dt_col: name of datetime column in the input data frame
        :target_col: name of target column in the input data frame
        :extra_features_col: name of extra feature columns that needs to predict the target column.
        :param drop_missing: whether to drop missing values in the curve, if this is set to False,
                             an error will be reported if missing values are found. If True, will
                             drop the missing values and won't throw errors.
        """
        # self.scaler = MinMaxScaler()
        self.scaler = StandardScaler()
        self.config = None
        self.dt_col = dt_col
        if isinstance(target_col, str):
            self.target_col = [target_col]
        else:
            self.target_col = target_col
        self.extra_features_col = extra_features_col
        self.feature_data = None
        self.drop_missing = drop_missing
        self.generate_feature_list = None
        self.past_seq_len = None
        self.future_seq_len = future_seq_len
        self.time_features = time_features

    def _fit_transform(self, input_df):
        """
        Fit data and transform the raw data to features. This is used in training for hyper
        parameter searching.
        This method will refresh the parameters (e.g. min and max of the MinMaxScaler) if any
        :param input_df: The input time series data frame, Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
            in the last dimension, the 1st col is the time index
            (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length)
            if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
            if future sequence length = 1
        """
        self._check_input(input_df, mode="train")
        # print(input_df.shape)
        from bigdl.nano.utils.log4Error import invalidInputError
        feature_data = self._get_features(input_df, self.config)
        self.scaler.fit(feature_data)
        data_n = self._scale(feature_data)
        invalidInputError(np.mean(data_n[0]) < 1e-5,
                          "data_n[0] mean should be 0")
        (x, y) = self._roll_train(data_n,
                                  past_seq_len=self.past_seq_len,
                                  future_seq_len=self.future_seq_len)

        return x, y

    def fit_transform(self, input_df, **config):
        """
        Fit data and transform the raw data to features. This is used in training for hyper
        parameter searching.
        This method will refresh the parameters (e.g. min and max of the MinMaxScaler) if any
        :param input_df: The input time series data frame, it can be a list of data frame or just
         one dataframe
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01 1.9 1   2
         2019-01-02 2.3 0   2
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
            in the last dimension, the 1st col is the time index
            (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length)
            if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
            if future sequence length = 1
        """
        self.config = self._get_feat_config(**config)

        if isinstance(input_df, list):
            train_x_list = []
            train_y_list = []
            for df in input_df:
                x, y = self._fit_transform(df)
                train_x_list.append(x)
                train_y_list.append(y)
            train_x = np.concatenate(train_x_list, axis=0)
            train_y = np.concatenate(train_y_list, axis=0)
        else:
            train_x, train_y = self._fit_transform(input_df)
        return train_x, train_y

    def _transform(self, input_df, mode):
        """
        Transform data into features using the preset of configurations from fit_transform
        :param input_df: The input time series data frame.
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01  1.9         1                       2
         2019-01-02  2.3         0                       2
        :param mode: 'val'/'test'.
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
            in the last dimension, the 1st col is the time index
            (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length)
            if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
            if future sequence length = 1
        """
        self._check_input(input_df, mode)
        # generate features
        feature_data = self._get_features(input_df, self.config)
        # select and standardize data
        data_n = self._scale(feature_data)
        if mode == 'val':
            (x, y) = self._roll_train(data_n,
                                      past_seq_len=self.past_seq_len,
                                      future_seq_len=self.future_seq_len)
            return x, y
        else:
            x = self._roll_test(data_n, past_seq_len=self.past_seq_len)
            return x, None

    def transform(self, input_df, is_train=True):
        """
        Transform data into features using the preset of configurations from fit_transform
        :param input_df: The input time series data frame, input_df can be a list of data frame or
                         one data frame.
         Example:
         datetime   value   "extra feature 1"   "extra feature 2"
         2019-01-01  1.9         1                       2
         2019-01-02  2.3         0                       2
        :param is_train: If the input_df is for training.
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length),
            in the last dimension, the 1st col is the time index
            (data type needs to be numpy datetime type, e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length)
            if future sequence length > 1, or 1-d numpy array in format (no. of samples, )
            if future sequence length = 1
        """
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.config is None or self.past_seq_len is None:
            invalidInputError(False,
                              "Needs to call fit_transform or restore"
                              " first before calling transform")
        mode = "val" if is_train else "test"
        if isinstance(input_df, list):
            output_x_list = []
            output_y_list = []
            for df in input_df:
                if mode == 'val':
                    x, y = self._transform(df, mode)
                    output_x_list.append(x)
                    output_y_list.append(y)
                else:
                    x, _ = self._transform(df, mode)
                    output_x_list.append(x)
            output_x = np.concatenate(output_x_list, axis=0)
            if output_y_list:
                output_y = np.concatenate(output_y_list, axis=0)
            else:
                output_y = None
        else:
            output_x, output_y = self._transform(input_df, mode)
        return output_x, output_y

    def _unscale(self, y):
        # for standard scalar
        y_unscale = np.zeros(y.shape)
        for i in range(len(self.target_col)):
            value_mean = self.scaler.mean_[i]
            value_scale = self.scaler.scale_[i]
            y_unscale[:, i:i+self.future_seq_len] = \
                y[:, i:i+self.future_seq_len] * value_scale + value_mean
        return y_unscale

    def unscale_uncertainty(self, y_uncertainty):
        y_uncertainty_unscale = np.zeros(y_uncertainty.shape)
        for i in range(len(self.target_col)):
            value_scale = self.scaler.scale_[i]
            if len(self.target_col) == 1:
                y_uncertainty_unscale = y_uncertainty * value_scale
            else:
                y_uncertainty_unscale[:, :, i] = y_uncertainty[:, :, i] * value_scale
        return y_uncertainty_unscale

    def _get_y_pred_df(self, y_pred_dt_df, y_pred_unscale):
        """
        get prediction data frame with datetime column and target column.
        :param input_df:
        :return : prediction data frame. If future_seq_len is 1, the output data frame columns are
            datetime | {target_col}. Otherwise, the output data frame columns are
            datetime | {target_col}_0 | {target_col}_1 | ...
        """
        y_pred_df = y_pred_dt_df
        if self.future_seq_len > 1:
            for i in range(self.future_seq_len):
                for j in range(len(self.target_col)):
                    column = self.target_col[j] + "_" + str(i)
                    y_pred_df[column] = pd.DataFrame(y_pred_unscale[:, i])
        else:
            y_pred_df[self.target_col] = pd.DataFrame(y_pred_unscale)
        return y_pred_df

    def post_processing(self, input_df, y_pred, is_train):
        """
        Used only in pipeline predict, after calling self.transform(input_df, is_train=False).
        Post_processing includes converting the predicted array into data frame and scalar inverse
        transform.
        :param input_df: a list of data frames or one data frame.
        :param y_pred: Model prediction result (ndarray).
        :param is_train: indicate the output is used to evaluation or prediction.
        :return:
         In validation mode (is_train=True), return the unscaled y_pred and rolled input_y.
         In test mode (is_train=False) return unscaled data frame(s) in the format of
          {datetime_col} | {target_col(s)}.
        """
        from bigdl.nano.utils.log4Error import invalidInputError
        y_pred_unscale = self._unscale(y_pred)
        if is_train:
            # return unscaled y_pred (ndarray) and y (ndarray).
            if isinstance(input_df, list):
                y_unscale_list = []
                for df in input_df:
                    _, y_unscale = self._roll_train(df[self.target_col],
                                                    self.past_seq_len,
                                                    self.future_seq_len)
                    y_unscale_list.append(y_unscale)
                output_y_unscale = np.concatenate(y_unscale_list, axis=0)
            else:
                _, output_y_unscale = self._roll_train(input_df[self.target_col],
                                                       self.past_seq_len,
                                                       self.future_seq_len)
            return output_y_unscale, y_pred_unscale

        else:
            # return data frame or a list of data frames.
            if isinstance(input_df, list):
                y_pred_dt_df_list = self._get_y_pred_dt_df(input_df, self.past_seq_len)
                y_pred_df_list = []
                y_pred_st_loc = 0
                for y_pred_dt_df in y_pred_dt_df_list:
                    df = self._get_y_pred_df(y_pred_dt_df,
                                             y_pred_unscale[y_pred_st_loc:
                                                            y_pred_st_loc + len(y_pred_dt_df)])
                    y_pred_st_loc = y_pred_st_loc + len(y_pred_dt_df)
                    y_pred_df_list.append(df)
                    invalidInputError(y_pred_st_loc == len(y_pred_unscale),
                                      "y_pred_st_loc should match len(y_pred_unscale)")
                return y_pred_df_list
            else:
                y_pred_dt_df = self._get_y_pred_dt_df(input_df, self.past_seq_len)
                y_pred_df = self._get_y_pred_df(y_pred_dt_df, y_pred_unscale)
                return y_pred_df

    def save(self, file_path, replace=False):
        """
        save the feature tools internal variables as well as the initialization args.
        Some of the variables are derived after fit_transform, so only saving config is not enough.
        :param: file : the file to be saved
        :return:
        """
        # for StandardScaler()
        data_to_save = {"mean": self.scaler.mean_.tolist(),
                        "scale": self.scaler.scale_.tolist(),
                        "future_seq_len": self.future_seq_len,
                        "dt_col": self.dt_col,
                        "target_col": self.target_col,
                        "extra_features_col": self.extra_features_col,
                        "drop_missing": self.drop_missing
                        }
        save_config(file_path, data_to_save, replace=replace)

    def restore(self, **config):
        """
        Restore variables from file
        :return:
        """
#         with open(file_path, 'r') as input_file:
#             result = json.load(input_file)

        # for StandardScalar()
        self.scaler = StandardScaler()
        self.scaler.mean_ = np.asarray(config["mean"])
        self.scaler.scale_ = np.asarray(config["scale"])

        self.config = self._get_feat_config(**config)

        self.future_seq_len = config["future_seq_len"]
        self.dt_col = config["dt_col"]
        self.target_col = config["target_col"]
        self.extra_features_col = config["extra_features_col"]
        self.drop_missing = config["drop_missing"]

        # for MinMaxScalar()
        # self.scaler = MinMaxScaler()
        # self.scaler.min_ = np.asarray(result["min"])
        # self.scaler.scale_ = np.asarray(result["scale"])
        # print(self.scaler.transform(input_data))

    def get_feature_list(self):
        feature_list = []
        if self.time_features:
            for feature in (TIME_FEATURE + ADDITIONAL_TIME_FEATURE):
                feature_list.append(feature + "({})".format(self.dt_col))
        if self.extra_features_col:
            feature_list += self.extra_features_col
        return feature_list

    def get_feature_dim(self):
        return len(self.get_feature_list()) + len(self.target_col)

    def get_target_dim(self):
        return len(self.target_col)

    def _get_feat_config(self, **config):
        """
        Get feature related arguments from global hyper parameter config and do necessary error
        checking
        :param config: the global config (usually from hyper parameter tuning)
        :return: config only for feature engineering
        """
        self._check_config(**config)
        feature_config_names = ["selected_features", "past_seq_len"]
        feat_config = {}
        for name in feature_config_names:
            if name not in config:
                continue
            feat_config[name] = config[name]
        self.past_seq_len = feat_config.get("past_seq_len", 2)
        return feat_config

    def _check_input(self, input_df, mode="train"):
        """
        Check dataframe for integrity. Requires time sequence to come in uniform sampling intervals.
        :param input_df:
        :return:
        """
        # check NaT in datetime
        input_df = input_df.reset_index()
        from bigdl.nano.utils.log4Error import invalidInputError
        dt = input_df[self.dt_col]
        if not np.issubdtype(dt, np.datetime64):
            invalidInputError(False,
                              "The dtype of datetime column is required to be np.datetime64!")
        is_nat = pd.isna(dt)
        if is_nat.any(axis=None):
            invalidInputError(False, "Missing datetime in input dataframe!")

        # check uniform (is that necessary?)
        interval = dt[1] - dt[0]

        if not all([dt[i] - dt[i - 1] == interval for i in range(1, len(dt))]):
            invalidInputError(False, "Input time sequence intervals are not uniform!")

        # check missing values
        if not self.drop_missing:
            is_nan = pd.isna(input_df)
            if is_nan.any(axis=None):
                invalidInputError(False, "Missing values in input dataframe!")

        # check if the length of input data is smaller than requested.
        if mode == "test":
            min_input_len = self.past_seq_len
            error_msg = "Length of {} data should be larger than " \
                        "the past sequence length selected by automl.\n" \
                        "{} data length: {}\n" \
                        "past sequence length selected: {}\n" \
                .format(mode, mode, len(input_df), self.past_seq_len)
        else:
            min_input_len = self.past_seq_len + self.future_seq_len
            error_msg = "Length of {} data should be larger than " \
                        "the sequence length you want to predict " \
                        "plus the past sequence length selected by automl.\n"\
                        "{} data length: {}\n"\
                        "predict sequence length: {}\n"\
                        "past sequence length selected: {}\n"\
                .format(mode, mode, len(input_df), self.future_seq_len, self.past_seq_len)
        if len(input_df) < min_input_len:
            invalidInputError(False, error_msg)

        return input_df

    def _roll_data(self, data, seq_len):
        result = []
        mask = []

        for i in range(len(data) - seq_len + 1):
            if seq_len == 1 and len(self.target_col) > 1:
                result.append(data[i])
            else:
                result.append(data[i: i + seq_len])

            if pd.isna(data[i: i + seq_len]).any(axis=None):
                mask.append(0)
            else:
                mask.append(1)

        return np.asarray(result), np.asarray(mask)

    def _roll_train(self, dataframe, past_seq_len, future_seq_len):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        roll_train: split the whole dataset apart to build (x, y).
        :param df: a dataframe which has been resampled in uniform frequency.
        :param past_seq_len: the length of the past sequence
        :param future_seq_len: the length of the future sequence
        :return: tuple (x,y)
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the
            last dimension, the 1st col is the time index (data type needs to be numpy datetime type
            , e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
            y: y is 2-d numpy array in format (no. of samples, future sequence length) if future
            sequence length > 1, or 1-d numpy array in format (no. of samples, ) if future sequence
            length = 1
        """
        x = dataframe[0:-future_seq_len].values
        if len(self.target_col) == 1:
            y = dataframe.iloc[past_seq_len:, 0].values
        else:
            y = dataframe.iloc[past_seq_len:, list(range(0, len(self.target_col)))].values
        output_x, mask_x = self._roll_data(x, past_seq_len)
        output_y, mask_y = self._roll_data(y, future_seq_len)
        # output_x.shape[0] == output_y.shape[0],
        # "The shape of output_x and output_y doesn't match! "
        mask = (mask_x == 1) & (mask_y == 1)
        return output_x[mask], output_y[mask]

    def _roll_test(self, dataframe, past_seq_len):
        """
        roll dataframe into sequence samples to be used in TimeSequencePredictor.
        roll_test: the whole dataframe is regarded as x.
        :param df: a dataframe which has been resampled in uniform frequency.
        :param past_seq_len: the length of the past sequence
        :return: x
            x: 3-d array in format (no. of samples, past sequence length, 2+feature length), in the
            last dimension, the 1st col is the time index (data type needs to be numpy datetime type
            , e.g. "datetime64"),
            the 2nd col is the target value (data type should be numeric)
        """
        x = dataframe.values
        output_x, mask_x = self._roll_data(x, past_seq_len)
        # output_x.shape[0] == output_y.shape[0],
        # "The shape of output_x and output_y doesn't match! "
        mask = (mask_x == 1)
        return output_x[mask]

    def __get_y_pred_dt_df(self, input_df, past_seq_len):
        """
        :param input_df: one data frame
        :return: a data frame with prediction datetime
        """
        input_df = input_df.reset_index(drop=True)
        input_dt_df = input_df.reset_index(drop=True)[[self.dt_col]].copy()
        time_delta = input_dt_df.iloc[-1] - input_dt_df.iloc[-2]
        last_time = input_dt_df.iloc[-1] + time_delta
        last_df = pd.DataFrame({self.dt_col: last_time})
        pre_pred_dt_df = input_dt_df[past_seq_len:].copy()
        pre_pred_dt_df = pre_pred_dt_df.reset_index(drop=True)
        y_pred_dt_df = pre_pred_dt_df.append(last_df, ignore_index=True)
        # print(y_pred_dt_df)
        return y_pred_dt_df

    def _get_y_pred_dt_df(self, input_df, past_seq_len):
        """
        :param input_df: a data frame or a list of data frame
        :param past_seq_len:
        :return:
        """
        if isinstance(input_df, list):
            y_pred_dt_df_list = []
            for df in input_df:
                y_pred_dt_df = self.__get_y_pred_dt_df(df, past_seq_len)
                y_pred_dt_df_list.append(y_pred_dt_df)
            return y_pred_dt_df_list
        else:
            return self.__get_y_pred_dt_df(input_df, past_seq_len)

    def _scale(self, data):
        """
        Scale the data
        :param data:
        :return:
        """
        # n_features_in_ only for 0.23 sklearn support, sklearn version >=0.24 will not check this
        if sklearn.__version__[:4] == "0.23":
            self.scaler.n_features_in_ = self.scaler.mean_.shape[0]
        np_scaled = self.scaler.transform(data)
        data_s = pd.DataFrame(np_scaled)
        return data_s

    def _rearrange_data(self, input_df):
        """
        change the input_df column order into [datetime, target, feature1, feature2, ...]
        :param input_df:
        :return:
        """
        cols = input_df.columns.tolist()
        new_cols = [self.dt_col] + self.target_col +\
                   [col for col in cols if col != self.dt_col and col not in self.target_col]
        rearranged_data = input_df[new_cols].copy
        return rearranged_data

    def _generate_features(self, input_df):
        df = input_df.copy()
        df["id"] = df.index + 1
        field = df[self.dt_col]

        # built in time features
        for attr in TIME_FEATURE:
            if attr == "WEEKOFYEAR" and \
                    version.parse(pd.__version__) >= version.parse("1.1.0"):
                # DatetimeProperties.weekofyear has been deprecated since pandas 1.1.0,
                # convert to DatetimeIndex to fix, and call pd.Int64Index to return a index
                field_datetime = pd.to_datetime(field.values.astype(np.int64))
                df[attr + "({})".format(self.dt_col)] =\
                    pd.Int64Index(field_datetime.isocalendar().week)
            else:
                df[attr + "({})".format(self.dt_col)] = getattr(field.dt, attr.lower())

        # additional time features
        hour = field.dt.hour
        weekday = field.dt.weekday
        df["IS_AWAKE" + "({})".format(self.dt_col)] =\
            (((hour >= 6) & (hour <= 23)) | (hour == 0)).astype(int).values
        df["IS_BUSY_HOURS" + "({})".format(self.dt_col)] =\
            (((hour >= 7) & (hour <= 9)) | (hour >= 16) & (hour <= 19)).astype(int).values
        df["IS_WEEKEND" + "({})".format(self.dt_col)] =\
            (weekday >= 5).values

        return df

    def _get_features(self, input_df, config):
        feature_matrix = self._generate_features(input_df)
        selected_features = config.get("selected_features")
        if selected_features:
            feature_cols = np.asarray(json.loads(selected_features))
        else:
            feature_cols = self.get_feature_list()
        # we do not include target col in candidates.
        # the first column is designed to be the default position of target column.
        target_col = np.array(self.target_col)
        cols = np.concatenate([target_col, feature_cols])
        target_feature_matrix = feature_matrix[cols]
        return target_feature_matrix.astype(float)

    def _get_optional_parameters(self):
        return {"past_seq_len", "selected_features"}

    def _get_required_parameters(self):
        return set()
