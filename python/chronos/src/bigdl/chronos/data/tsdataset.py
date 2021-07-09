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

import pandas as pd
import numpy as np
import functools

from zoo.chronos.data.utils.feature import generate_dt_features, generate_global_features
from zoo.chronos.data.utils.impute import impute_timeseries_dataframe
from zoo.chronos.data.utils.deduplicate import deduplicate_timeseries_dataframe
from zoo.chronos.data.utils.roll import roll_timeseries_dataframe
from zoo.chronos.data.utils.scale import unscale_timeseries_numpy
from zoo.chronos.data.utils.resample import resample_timeseries_dataframe
from zoo.chronos.data.utils.split import split_timeseries_dataframe

from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.utilities.dataframe_functions import impute as impute_tsfresh
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters,\
    MinimalFCParameters, EfficientFCParameters
DEFAULT_PARAMS = {"comprehensive": ComprehensiveFCParameters(),
                  "minimal": MinimalFCParameters(),
                  "efficient": EfficientFCParameters()}

_DEFAULT_ID_COL_NAME = "id"
_DEFAULT_ID_PLACEHOLDER = "0"


class TSDataset:
    def __init__(self, data, **schema):
        '''
        TSDataset is an abstract of time series dataset.
        Cascade call is supported for most of the transform methods.
        '''
        self.df = data
        self.id_col = schema["id_col"]
        self.dt_col = schema["dt_col"]
        self.feature_col = schema["feature_col"].copy()
        self.target_col = schema["target_col"].copy()

        self.numpy_x = None
        self.numpy_y = None
        self.roll_feature = None
        self.roll_target = None
        self.roll_feature_df = None
        self.roll_addional_feature = None
        self.scaler = None
        self.scaler_index = [i for i in range(len(self.target_col))]
        self.id_sensitive = None

        self._check_basic_invariants()

        self._id_list = list(np.unique(self.df[self.id_col]))
        self._is_pd_datetime = pd.api.types.is_datetime64_any_dtype(self.df[self.dt_col].dtypes)

    @staticmethod
    def from_pandas(df,
                    dt_col,
                    target_col,
                    id_col=None,
                    extra_feature_col=None,
                    with_split=False,
                    val_ratio=0,
                    test_ratio=0.1,
                    largest_look_back=0,
                    largest_horizon=1):
        '''
        Initialize tsdataset(s) from pandas dataframe.

        :param df: a pandas dataframe for your raw time series data.
        :param dt_col: a str indicates the col name of datetime
               column in the input data frame.
        :param target_col: a str or list indicates the col name of target column
               in the input data frame.
        :param id_col: (optional) a str indicates the col name of dataframe id. If
               it is not explicitly stated, then the data is interpreted as only
               containing a single id.
        :param extra_feature_col: (optional) a str or list indicates the col name
               of extra feature columns that needs to predict the target column.
        :param with_split: (optional) bool, states if we need to split the dataframe
               to train, validation and test set. The value defaults to False.
        :param val_ratio: (optional) float, validation ratio. Only effective when
               with_split is set to True. The value defaults to 0.
        :param test_ratio: (optional) float, test ratio. Only effective when with_split
               is set to True. The value defaults to 0.1.
        :param largest_look_back: (optional) int, the largest length to look back.
               Only effective when with_split is set to True. The value defaults to 0.
        :param largest_horizon: (optional) int, the largest num of steps to look
               forward. Only effective when with_split is set to True. The value defaults
               to 1.

        :return: a TSDataset instance when with_split is set to False,
                 three TSDataset instances when with_split is set to True.

        Create a tsdataset instance by:

        >>> # Here is a df example:
        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"
        >>> # 00        2019-01-01    1.9     1                   2
        >>> # 01        2019-01-01    2.3     0                   9
        >>> # 00        2019-01-02    2.4     3                   4
        >>> # 01        2019-01-02    2.6     0                   2
        >>> tsdataset = TSDataset.from_pandas(df, dt_col="datetime",
        >>>                                   target_col="value", id_col="id",
        >>>                                   extra_feature_col=["extra feature 1",
        >>>                                                      "extra feature 2"])
        '''

        _check_type(df, "df", pd.DataFrame)

        tsdataset_df = df.copy(deep=True)
        target_col = _to_list(target_col, name="target_col")
        feature_col = _to_list(extra_feature_col, name="extra_feature_col")

        if id_col is None:
            tsdataset_df[_DEFAULT_ID_COL_NAME] = _DEFAULT_ID_PLACEHOLDER
            id_col = _DEFAULT_ID_COL_NAME

        if with_split:
            tsdataset_dfs = split_timeseries_dataframe(df=tsdataset_df,
                                                       id_col=id_col,
                                                       val_ratio=val_ratio,
                                                       test_ratio=test_ratio,
                                                       look_back=largest_look_back,
                                                       horizon=largest_horizon)
            return [TSDataset(data=tsdataset_dfs[i],
                              id_col=id_col,
                              dt_col=dt_col,
                              target_col=target_col,
                              feature_col=feature_col) for i in range(3)]

        return TSDataset(data=tsdataset_df,
                         id_col=id_col,
                         dt_col=dt_col,
                         target_col=target_col,
                         feature_col=feature_col)

    def impute(self, mode="last", const_num=0):
        '''
        Impute the tsdataset by imputing each univariate time series
        distinguished by id_col and feature_col.

        :param mode: imputation mode, select from "last", "const" or "linear".

            "last": impute by propagating the last non N/A number to its following N/A.
            if there is no non N/A number ahead, 0 is filled instead.

            "const": impute by a const value input by user.

            "linear": impute by linear interpolation.
        :param const_num:  indicates the const number to fill, which is only effective when mode
            is set to "const".

        :return: the tsdataset instance.
        '''
        df_list = [impute_timeseries_dataframe(df=self.df[self.df[self.id_col] == id_name],
                                               dt_col=self.dt_col,
                                               mode=mode,
                                               const_num=const_num)
                   for id_name in self._id_list]
        self.df = pd.concat(df_list)
        return self

    def deduplicate(self):
        '''
        Remove those duplicated records which has exactly the same values in each feature_col
        for each multivariate timeseries distinguished by id_col.

        :return: the tsdataset instance.
        '''
        df_list = [deduplicate_timeseries_dataframe(df=self.df[self.df[self.id_col] == id_name],
                                                    dt_col=self.dt_col)
                   for id_name in self._id_list]
        self.df = pd.concat(df_list)
        return self

    def resample(self, interval, start_time=None, end_time=None, merge_mode="mean"):
        '''
        Resample on a new interval for each univariate time series distinguished
        by id_col and feature_col.

        :param interval: pandas offset aliases, indicating time interval of the output dataframe.
        :param start_time: start time of the output dataframe.
        :param end_time: end time of the output dataframe.
        :param merge_mode: if current interval is smaller than output interval,
            we need to merge the values in a mode. "max", "min", "mean"
            or "sum" are supported for now.

        :return: the tsdataset instance.
        '''
        df_list = []
        for id_name in self._id_list:
            df_id = resample_timeseries_dataframe(df=self.df[self.df[self.id_col] == id_name]
                                                  .drop(self.id_col, axis=1),
                                                  dt_col=self.dt_col,
                                                  interval=interval,
                                                  start_time=start_time,
                                                  end_time=end_time,
                                                  merge_mode=merge_mode)
            df_id[self.id_col] = id_name
            df_list.append(df_id.copy())
        self.df = pd.concat(df_list)
        return self

    def gen_dt_feature(self):
        '''
        | Generate datetime feature for each row. Currently we generate following features:
        | "MINUTE": The minute of the time stamp.
        | "DAY": The day of the time stamp.
        | "DAYOFYEAR": The ordinal day of the year of the time stamp.
        | "HOUR": The hour of the time stamp.
        | "WEEKDAY": The day of the week of the time stamp, Monday=0, Sunday=6.
        | "WEEKOFYEAR": The ordinal week of the year of the time stamp.
        | "MONTH": The month of the time stamp.
        | "IS_AWAKE": Bool value indicating whether it belongs to awake hours for the time stamp,
        | True for hours between 6A.M. and 1A.M.
        | "IS_BUSY_HOURS": Bool value indicating whether it belongs to busy hours for the time
        | stamp, True for hours between 7A.M. and 10A.M. and hours between 4P.M. and 8P.M.
        | "IS_WEEKEND": Bool value indicating whether it belongs to weekends for the time stamp,
        | True for Saturdays and Sundays.

        :return: the tsdataset instance.
        '''
        df_list = [generate_dt_features(input_df=self.df[self.df[self.id_col] == id_name],
                                        dt_col=self.dt_col)
                   for id_name in self._id_list]
        self.df = pd.concat(df_list)
        from zoo.chronos.data.utils.feature import TIME_FEATURE, \
            ADDITIONAL_TIME_FEATURE_HOUR, ADDITIONAL_TIME_FEATURE_WEEKDAY
        increased_attrbutes = list(TIME_FEATURE) +\
            list(ADDITIONAL_TIME_FEATURE_HOUR) +\
            list(ADDITIONAL_TIME_FEATURE_WEEKDAY)
        self.feature_col += [attr + "({})".format(self.dt_col) for attr in increased_attrbutes]
        return self

    def gen_global_feature(self, settings="comprehensive", full_settings=None):
        '''
        Generate per-time-series feature for each time series.
        This method will be implemented by tsfresh.

        TODO: relationship with scale should be figured out.

        :param settings: str or dict. If a string is set, then it must be one of "comprehensive"
               "minimal" and "efficient". If a dict is set, then it should follow the instruction
               for default_fc_parameters in tsfresh. The value is defaulted to "comprehensive".
        :param full_settings: dict. It should follow the instruction for kind_to_fc_parameters in
               tsfresh. The value is defaulted to None.

        :return: the tsdataset instance.

        '''
        if full_settings is not None:
            self.df,\
                addtional_feature =\
                generate_global_features(input_df=self.df,
                                         column_id=self.id_col,
                                         column_sort=self.dt_col,
                                         kind_to_fc_parameters=full_settings)
            self.feature_col += addtional_feature
            return self

        if isinstance(settings, str):
            assert settings in ["comprehensive", "minimal", "efficient"], \
                f"settings str should be one of \"comprehensive\", \"minimal\", \"efficient\"\
                    , but found {settings}."
            default_fc_parameters = DEFAULT_PARAMS[settings]
        else:
            default_fc_parameters = settings

        self.df,\
            addtional_feature =\
            generate_global_features(input_df=self.df,
                                     column_id=self.id_col,
                                     column_sort=self.dt_col,
                                     default_fc_parameters=default_fc_parameters)

        self.feature_col += addtional_feature

        return self

    def gen_rolling_feature(self,
                            window_size,
                            settings="comprehensive",
                            full_settings=None):
        '''
        Generate aggregation feature for each sample.
        This method will be implemented by tsfresh.

        TODO: relationship with scale should be figured out.

        :param window_size: int, generate feature according to the rolling result.
        :param settings: str or dict. If a string is set, then it must be one of "comprehensive"
               "minimal" and "efficient". If a dict is set, then it should follow the instruction
               for default_fc_parameters in tsfresh. The value is defaulted to "comprehensive".
        :param full_settings: dict. It should follow the instruction for kind_to_fc_parameters in
               tsfresh. The value is defaulted to None.

        :return: the tsdataset instance.
        '''
        if isinstance(settings, str):
            assert settings in ["comprehensive", "minimal", "efficient"], \
                f"settings str should be one of \"comprehensive\", \"minimal\", \"efficient\"\
                    , but found {settings}."
            default_fc_parameters = DEFAULT_PARAMS[settings]
        else:
            default_fc_parameters = settings

        df_rolled = roll_time_series(self.df,
                                     column_id=self.id_col,
                                     column_sort=self.dt_col,
                                     max_timeshift=window_size-1,
                                     min_timeshift=window_size-1)
        if not full_settings:
            self.roll_feature_df = extract_features(df_rolled,
                                                    column_id=self.id_col,
                                                    column_sort=self.dt_col,
                                                    default_fc_parameters=default_fc_parameters)
        else:
            self.roll_feature_df = extract_features(df_rolled,
                                                    column_id=self.id_col,
                                                    column_sort=self.dt_col,
                                                    kind_to_fc_parameters=full_settings)
        impute_tsfresh(self.roll_feature_df)

        self.feature_col += list(self.roll_feature_df.columns)
        self.roll_addional_feature = list(self.roll_feature_df.columns)

        return self

    def roll(self,
             lookback,
             horizon,
             feature_col=None,
             target_col=None,
             id_sensitive=False):
        '''
        Sampling by rolling for machine learning/deep learning models.

        :param lookback: int, lookback value.
        :param horizon: int or list,
               if `horizon` is an int, we will sample `horizon` step
               continuously after the forecasting point.
               if `horizon` is a list, we will sample discretely according
               to the input list.
               specially, when `horizon` is set to 0, ground truth will be generated as None.
        :param feature_col: str or list, indicates the feature col name. Default to None,
               where we will take all available feature in rolling.
        :param target_col: str or list, indicates the target col name. Default to None,
               where we will take all target in rolling. it should be a subset of target_col
               you used to initialize the tsdataset.
        :param id_sensitive: bool,
               if `id_sensitive` is False, we will rolling on each id's sub dataframe
               and fuse the sampings.
               The shape of rolling will be
               x: (num_sample, lookback, num_feature_col + num_target_col)
               y: (num_sample, horizon, num_target_col)
               where num_sample is the summation of sample number of each dataframe

               if `id_sensitive` is True, we will rolling on the wide dataframe whose
               columns are cartesian product of id_col and feature_col
               The shape of rolling will be
               x: (num_sample, lookback, new_num_feature_col + new_num_target_col)
               y: (num_sample, horizon, new_num_target_col)
               where num_sample is the sample number of the wide dataframe,
               new_num_feature_col is the product of the number of id and the number of feature_col.
               new_num_target_col is the product of the number of id and the number of target_col.

        :return: the tsdataset instance.

        roll() can be called by:

        >>> # Here is a df example:
        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"
        >>> # 00        2019-01-01    1.9     1                   2
        >>> # 01        2019-01-01    2.3     0                   9
        >>> # 00        2019-01-02    2.4     3                   4
        >>> # 01        2019-01-02    2.6     0                   2
        >>> tsdataset = TSDataset.from_pandas(df, dt_col="datetime",
        >>>                                   target_col="value", id_col="id",
        >>>                                   extra_feature_col=["extra feature 1",
        >>>                                                      "extra feature 2"])
        >>> horizon, lookback = 1, 1
        >>> tsdataset.roll(lookback=lookback, horizon=horizon, id_sensitive=False)
        >>> x, y = tsdataset.to_numpy()
        >>> print(x, y) # x = [[[1.9, 1, 2 ]], [[2.3, 0, 9 ]]] y = [[[ 2.4 ]], [[ 2.6 ]]]
        >>> print(x.shape, y.shape) # x.shape = (2, 1, 3) y.shape = (2, 1, 1)
        >>> tsdataset.roll(lookback=lookback, horizon=horizon, id_sensitive=True)
        >>> x, y = tsdataset.to_numpy()
        >>> print(x, y) # x = [[[ 1.9, 2.3, 1, 2, 0, 9 ]]] y = [[[ 2.4, 2.6]]]
        >>> print(x.shape, y.shape) # x.shape = (1, 1, 6) y.shape = (1, 1, 2)

        '''
        feature_col = _to_list(feature_col, "feature_col") if feature_col is not None \
            else self.feature_col
        target_col = _to_list(target_col, "target_col") if target_col is not None \
            else self.target_col
        if self.roll_addional_feature:
            additional_feature_col =\
                list(set(feature_col).intersection(set(self.roll_addional_feature)))
            feature_col =\
                list(set(feature_col) - set(self.roll_addional_feature))
            self.roll_feature = feature_col + additional_feature_col
        else:
            additional_feature_col = None
            self.roll_feature = feature_col

        self.roll_target = target_col
        num_id = len(self._id_list)
        num_feature_col = len(self.roll_feature)
        num_target_col = len(self.roll_target)
        self.id_sensitive = id_sensitive
        roll_feature_df = None if self.roll_feature_df is None \
            else self.roll_feature_df[additional_feature_col]

        rolling_result =\
            self.df.groupby([self.id_col])\
                   .apply(lambda df: roll_timeseries_dataframe(df=df,
                                                               roll_feature_df=roll_feature_df,
                                                               lookback=lookback,
                                                               horizon=horizon,
                                                               feature_col=feature_col,
                                                               target_col=target_col))

        # concat the result on required axis
        concat_axis = 2 if id_sensitive else 0
        self.numpy_x = np.concatenate([rolling_result[i][0]
                                       for i in self._id_list],
                                      axis=concat_axis).astype(np.float64)
        if horizon != 0:
            self.numpy_y = np.concatenate([rolling_result[i][1]
                                           for i in self._id_list],
                                          axis=concat_axis).astype(np.float64)
        else:
            self.numpy_y = None

        # target first
        if self.id_sensitive:
            feature_start_idx = num_target_col*num_id
            reindex_list = [list(range(i*num_target_col, (i+1)*num_target_col)) +
                            list(range(feature_start_idx+i*num_feature_col,
                                       feature_start_idx+(i+1)*num_feature_col))
                            for i in range(num_id)]
            reindex_list = functools.reduce(lambda a, b: a+b, reindex_list)
            sorted_index = sorted(range(len(reindex_list)), key=reindex_list.__getitem__)
            self.numpy_x = self.numpy_x[:, :, sorted_index]

        # scaler index
        num_roll_target = len(self.roll_target)
        repeat_factor = len(self._id_list) if self.id_sensitive else 1
        scaler_index = [self.target_col.index(self.roll_target[i])
                        for i in range(num_roll_target)] * repeat_factor
        self.scaler_index = scaler_index

        return self

    def to_numpy(self):
        '''
        Export rolling result in form of a tuple of numpy ndarray (x, y).

        :return: a 2-dim tuple. each item is a 3d numpy ndarray. The ndarray
                 is casted to float64.
        '''
        if self.numpy_x is None:
            raise RuntimeError("Please call \"roll\" method\
                    before transform a TSDataset to numpy ndarray!")
        return self.numpy_x, self.numpy_y

    def to_pandas(self):
        '''
        Export the pandas dataframe.

        :return: the internal dataframe.
        '''
        return self.df.copy()

    def scale(self, scaler, fit=True):
        '''
        Scale the time series dataset's feature column and target column.

        :param scaler: sklearn scaler instance, StandardScaler, MaxAbsScaler,
               MinMaxScaler and RobustScaler are supported.
        :param fit: if we need to fit the scaler. Typically, the value should
               be set to True for training set, while False for validation and
               test set. The value is defaulted to True.

        :return: the tsdataset instance.

        Assume there is a training set tsdata and a test set tsdata_test.
        scale() should be called first on training set with default value fit=True,
        then be called on test set with the same scaler and fit=False.

        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = StandardScaler()
        >>> tsdata.scale(scaler, fit=True)
        >>> tsdata_test.scale(scaler, fit=False)
        '''
        feature_col = self.feature_col
        if self.roll_addional_feature:
            feature_col = []
            for feature in self.feature_col:
                if feature not in self.roll_addional_feature:
                    feature_col.append(feature)
        if fit:
            self.df[self.target_col + feature_col] = \
                scaler.fit_transform(self.df[self.target_col + feature_col])
        else:
            self.df[self.target_col + feature_col] = \
                scaler.transform(self.df[self.target_col + feature_col])
        self.scaler = scaler
        return self

    def unscale(self):
        '''
        Unscale the time series dataset's feature column and target column.

        :return: the tsdataset instance.
        '''
        feature_col = self.feature_col
        if self.roll_addional_feature:
            feature_col = []
            for feature in self.feature_col:
                if feature not in self.roll_addional_feature:
                    feature_col.append(feature)
        self.df[self.target_col + feature_col] = \
            self.scaler.inverse_transform(self.df[self.target_col + feature_col])
        return self

    def unscale_numpy(self, data):
        '''
        Unscale the time series forecaster's numpy prediction result/ground truth.

        :param data: a numpy ndarray with 3 dim whose shape should be exactly the
               same with self.numpy_y.

        :return: the unscaled numpy ndarray.
        '''
        return unscale_timeseries_numpy(data, self.scaler, self.scaler_index)

    def _check_basic_invariants(self):
        '''
        This function contains a bunch of assertions to make sure strict rules(the invariants)
        for the internal dataframe(self.df) must stands. If not, clear and user-friendly error
        or warning message should be provided to the users.
        This function will be called after each method(e.g. impute, deduplicate ...).
        '''
        # check type
        _check_type(self.df, "df", pd.DataFrame)
        _check_type(self.id_col, "id_col", str)
        _check_type(self.dt_col, "dt_col", str)
        _check_type(self.target_col, "target_col", list)
        _check_type(self.feature_col, "feature_col", list)

        # check valid name
        _check_col_within(self.df, self.id_col)
        _check_col_within(self.df, self.dt_col)
        for target_col_name in self.target_col:
            _check_col_within(self.df, target_col_name)
        for feature_col_name in self.feature_col:
            if self.roll_addional_feature and feature_col_name in self.roll_addional_feature:
                continue
            _check_col_within(self.df, feature_col_name)

        # check no n/a in critical col
        _check_col_no_na(self.df, self.dt_col)
        _check_col_no_na(self.df, self.id_col)


def _to_list(item, name, expect_type=str):
    if isinstance(item, list):
        return item
    if item is None:
        return []
    _check_type(item, name, expect_type)
    return [item]


def _check_type(item, name, expect_type):
    assert isinstance(item, expect_type),\
        f"a {str(expect_type)} is expected for {name} but found {type(item)}"


def _check_col_within(df, col_name):
    assert col_name in df.columns,\
        f"{col_name} is expected in dataframe while not found"


def _check_col_no_na(df, col_name):
    _check_col_within(df, col_name)
    assert df[col_name].isna().sum() == 0,\
        f"{col_name} column should not have N/A."
