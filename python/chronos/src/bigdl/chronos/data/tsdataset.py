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

import pandas as pd
import numpy as np
import functools

from bigdl.chronos.data.utils.feature import generate_dt_features, generate_global_features
from bigdl.chronos.data.utils.impute import impute_timeseries_dataframe
from bigdl.chronos.data.utils.deduplicate import deduplicate_timeseries_dataframe
from bigdl.chronos.data.utils.roll import roll_timeseries_dataframe
from bigdl.chronos.data.utils.scale import unscale_timeseries_numpy
from bigdl.chronos.data.utils.resample import resample_timeseries_dataframe
from bigdl.chronos.data.utils.split import split_timeseries_dataframe
from bigdl.chronos.data.utils.cycle_detection import cycle_length_est
from bigdl.chronos.data.utils.utils import _to_list, _check_type,\
    _check_col_within, _check_col_no_na, _check_is_aligned, _check_dt_is_sorted

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
        self.roll_additional_feature = None
        self.scaler = None
        self.scaler_index = [i for i in range(len(self.target_col))]
        self.id_sensitive = None
        self._has_generate_agg_feature = False
        self._check_basic_invariants()

        self._id_list = list(np.unique(self.df[self.id_col]))
        self._freq_certainty = False
        self._freq = None
        self._is_pd_datetime = pd.api.types.is_datetime64_any_dtype(self.df[self.dt_col].dtypes)
        if self._is_pd_datetime:
            if len(self.df[self.dt_col]) < 2:
                self._freq = None
            else:
                self._freq = self.df[self.dt_col].iloc[1] - self.df[self.dt_col].iloc[0]

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
               column in the input data frame, the dt_col must be sorted
               from past to latest respectively for each id.
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

    @staticmethod
    def from_parquet(path,
                     dt_col,
                     target_col,
                     id_col=None,
                     extra_feature_col=None,
                     with_split=False,
                     val_ratio=0,
                     test_ratio=0.1,
                     largest_look_back=0,
                     largest_horizon=1,
                     **kwargs):
        """
        Initialize tsdataset(s) from path of parquet file.

        :param path: A string path to parquet file. The string could be a URL.
               Valid URL schemes include hdfs, http, ftp, s3, gs, and file. For file URLs, a host
               is expected. A local file could be: file://localhost/path/to/table.parquet.
               A file URL can also be a path to a directory that contains multiple partitioned
               parquet files.
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
        :param kwargs: Any additional kwargs are passed to the pd.read_parquet
               and pyarrow.parquet.read_table.

        :return: a TSDataset instance when with_split is set to False,
                 three TSDataset instances when with_split is set to True.

        Create a tsdataset instance by:

        >>> # Here is a df example:
        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"
        >>> # 00        2019-01-01    1.9     1                   2
        >>> # 01        2019-01-01    2.3     0                   9
        >>> # 00        2019-01-02    2.4     3                   4
        >>> # 01        2019-01-02    2.6     0                   2
        >>> tsdataset = TSDataset.from_parquet("hdfs://path/to/table.parquet", dt_col="datetime",
        >>>                                   target_col="value", id_col="id",
        >>>                                   extra_feature_col=["extra feature 1",
        >>>                                                      "extra feature 2"])
        """
        from bigdl.chronos.data.utils.file import parquet2pd
        columns = _to_list(dt_col, name="dt_col") + \
            _to_list(target_col, name="target_col") + \
            _to_list(id_col, name="id_col") + \
            _to_list(extra_feature_col, name="extra_feature_col")
        df = parquet2pd(path, columns=columns, **kwargs)
        return TSDataset.from_pandas(df,
                                     dt_col=dt_col,
                                     target_col=target_col,
                                     id_col=id_col,
                                     extra_feature_col=extra_feature_col,
                                     with_split=with_split,
                                     val_ratio=val_ratio,
                                     test_ratio=test_ratio,
                                     largest_look_back=largest_look_back,
                                     largest_horizon=largest_horizon,
                                     )

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
        self.df = self.df.groupby([self.id_col]) \
            .apply(lambda df: impute_timeseries_dataframe(df=df,
                                                          dt_col=self.dt_col,
                                                          mode=mode,
                                                          const_num=const_num))
        self.df.reset_index(drop=True, inplace=True)
        return self

    def deduplicate(self):
        '''
        Remove those duplicated records which has exactly the same values in each feature_col
        for each multivariate timeseries distinguished by id_col.

        :return: the tsdataset instance.
        '''
        self.df = deduplicate_timeseries_dataframe(df=self.df, dt_col=self.dt_col)
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
        assert self._is_pd_datetime,\
            "The time series data does not have a Pandas datetime format "\
            "(you can use pandas.to_datetime to convert a string into a datetime format)."
        from pandas.api.types import is_numeric_dtype
        type_error_list = [val for val in self.target_col + self.feature_col
                           if not is_numeric_dtype(self.df[val])]
        try:
            for val in type_error_list:
                self.df[val] = self.df[val].astype(np.float32)
        except Exception:
            raise RuntimeError("All the columns of target_col "
                               "and extra_feature_col should be of numeric type.")
        self.df = self.df.groupby([self.id_col]) \
            .apply(lambda df: resample_timeseries_dataframe(df=df,
                                                            dt_col=self.dt_col,
                                                            interval=interval,
                                                            start_time=start_time,
                                                            end_time=end_time,
                                                            id_col=self.id_col,
                                                            merge_mode=merge_mode))
        self._freq = pd.Timedelta(interval)
        self._freq_certainty = True
        self.df.reset_index(drop=True, inplace=True)
        return self

    def gen_dt_feature(self, features="auto", one_hot_features=None):
        '''
        Generate datetime feature(s) for each record.

        :param features: str or list, states which feature(s) will be generated. If the value
               is set to be a str, it should be one of "auto" or "all". For "auto", a subset
               of datetime features will be generated under the consideration of the sampling
               frequency of your data. For "all", the whole set of datetime features will be
               generated. If the value is set to be a list, the list should contain the features
               you want to generate. A table of all datatime features and their description is
               listed below. The value defaults to "auto".
        :param one_hot_features: list, states which feature(s) will be generated as one-hot-encoded
               feature. The value defaults to None, which means no features will be generated with\
               one-hot-encoded.

        | "MINUTE": The minute of the time stamp.
        | "DAY": The day of the time stamp.
        | "DAYOFYEAR": The ordinal day of the year of the time stamp.
        | "HOUR": The hour of the time stamp.
        | "WEEKDAY": The day of the week of the time stamp, Monday=0, Sunday=6.
        | "WEEKOFYEAR": The ordinal week of the year of the time stamp.
        | "MONTH": The month of the time stamp.
        | "YEAR": The year of the time stamp.
        | "IS_AWAKE": Bool value indicating whether it belongs to awake hours for the time stamp,
        | True for hours between 6A.M. and 1A.M.
        | "IS_BUSY_HOURS": Bool value indicating whether it belongs to busy hours for the time
        | stamp, True for hours between 7A.M. and 10A.M. and hours between 4P.M. and 8P.M.
        | "IS_WEEKEND": Bool value indicating whether it belongs to weekends for the time stamp,
        | True for Saturdays and Sundays.

        :return: the tsdataset instance.
        '''
        assert self._is_pd_datetime, "The time series data does not have a Pandas datetime format"\
            "(you can use pandas.to_datetime to convert a string into a datetime format.)"
        features_generated = []
        self.df = generate_dt_features(input_df=self.df,
                                       dt_col=self.dt_col,
                                       features=features,
                                       one_hot_features=one_hot_features,
                                       freq=self._freq,
                                       features_generated=features_generated)
        self.feature_col += features_generated
        return self

    def gen_global_feature(self, settings="comprehensive", full_settings=None, n_jobs=1):
        '''
        Generate per-time-series feature for each time series.
        This method will be implemented by tsfresh.
        Make sure that the specified column name does not contain '__'.

        TODO: relationship with scale should be figured out.

        :param settings: str or dict. If a string is set, then it must be one of "comprehensive"
               "minimal" and "efficient". If a dict is set, then it should follow the instruction
               for default_fc_parameters in tsfresh. The value is defaulted to "comprehensive".
        :param full_settings: dict. It should follow the instruction for kind_to_fc_parameters in
               tsfresh. The value is defaulted to None.
        :param n_jobs: int. The number of processes to use for parallelization.

        :return: the tsdataset instance.
        '''
        try:
            from tsfresh import extract_features
            from tsfresh.feature_extraction import ComprehensiveFCParameters, \
                MinimalFCParameters, EfficientFCParameters
        except ImportError:
            raise ImportError("Please install tsfresh by `pip install tsfresh` to use "
                              "`gen_global_feature` method.")

        DEFAULT_PARAMS = {"comprehensive": ComprehensiveFCParameters(),
                          "minimal": MinimalFCParameters(),
                          "efficient": EfficientFCParameters()}

        assert not self._has_generate_agg_feature, \
            "Only one of gen_global_feature and gen_rolling_feature should be called."
        if full_settings is not None:
            self.df,\
                addtional_feature =\
                generate_global_features(input_df=self.df,
                                         column_id=self.id_col,
                                         column_sort=self.dt_col,
                                         kind_to_fc_parameters=full_settings,
                                         n_jobs=n_jobs)
            self.feature_col += addtional_feature
            return self

        if isinstance(settings, str):
            assert settings in ['comprehensive', 'minimal', 'efficient'], \
                "settings str should be one of 'comprehensive', 'minimal', 'efficient'"\
                f", but found {settings}."
            default_fc_parameters = DEFAULT_PARAMS[settings]
        else:
            default_fc_parameters = settings

        self.df,\
            addtional_feature =\
            generate_global_features(input_df=self.df,
                                     column_id=self.id_col,
                                     column_sort=self.dt_col,
                                     default_fc_parameters=default_fc_parameters,
                                     n_jobs=n_jobs)

        self.feature_col += addtional_feature
        self._has_generate_agg_feature = True
        return self

    def gen_rolling_feature(self,
                            window_size,
                            settings="comprehensive",
                            full_settings=None,
                            n_jobs=1):
        '''
        Generate aggregation feature for each sample.
        This method will be implemented by tsfresh.
        Make sure that the specified column name does not contain '__'.

        TODO: relationship with scale should be figured out.

        :param window_size: int, generate feature according to the rolling result.
        :param settings: str or dict. If a string is set, then it must be one of "comprehensive"
               "minimal" and "efficient". If a dict is set, then it should follow the instruction
               for default_fc_parameters in tsfresh. The value is defaulted to "comprehensive".
        :param full_settings: dict. It should follow the instruction for kind_to_fc_parameters in
               tsfresh. The value is defaulted to None.
        :param n_jobs: int. The number of processes to use for parallelization.

        :return: the tsdataset instance.
        '''
        try:
            from tsfresh.utilities.dataframe_functions import roll_time_series
            from tsfresh.utilities.dataframe_functions import impute as impute_tsfresh
            from tsfresh import extract_features
            from tsfresh.feature_extraction import ComprehensiveFCParameters, \
                MinimalFCParameters, EfficientFCParameters
        except ImportError:
            raise ImportError("Please install tsfresh by `pip install tsfresh` to use "
                              "`gen_rolling_feature` method.")

        DEFAULT_PARAMS = {"comprehensive": ComprehensiveFCParameters(),
                          "minimal": MinimalFCParameters(),
                          "efficient": EfficientFCParameters()}

        assert not self._has_generate_agg_feature,\
            "Only one of gen_global_feature and gen_rolling_feature should be called."
        if isinstance(settings, str):
            assert settings in ['comprehensive', 'minimal', 'efficient'], \
                "settings str should be one of 'comprehensive', 'minimal', 'efficient'"\
                f", but found {settings}."
            default_fc_parameters = DEFAULT_PARAMS[settings]
        else:
            default_fc_parameters = settings

        assert window_size < self.df.groupby(self.id_col).size().min() + 1, "gen_rolling_feature "\
            "should have a window_size smaller than shortest time series length."
        df_rolled = roll_time_series(self.df,
                                     column_id=self.id_col,
                                     column_sort=self.dt_col,
                                     max_timeshift=window_size - 1,
                                     min_timeshift=window_size - 1,
                                     n_jobs=n_jobs)
        if not full_settings:
            self.roll_feature_df = extract_features(df_rolled,
                                                    column_id=self.id_col,
                                                    column_sort=self.dt_col,
                                                    default_fc_parameters=default_fc_parameters,
                                                    n_jobs=n_jobs)
        else:
            self.roll_feature_df = extract_features(df_rolled,
                                                    column_id=self.id_col,
                                                    column_sort=self.dt_col,
                                                    kind_to_fc_parameters=full_settings,
                                                    n_jobs=n_jobs)
        impute_tsfresh(self.roll_feature_df)

        self.feature_col += list(self.roll_feature_df.columns)
        self.roll_additional_feature = list(self.roll_feature_df.columns)
        self._has_generate_agg_feature = True
        return self

    def roll(self,
             horizon,
             lookback='auto',
             feature_col=None,
             target_col=None,
             id_sensitive=False):
        '''
        Sampling by rolling for machine learning/deep learning models.

        :param lookback: int, lookback value. Default to 'auto',
               if 'auto', the mode of time series' cycle length will be taken as the lookback.
        :param horizon: int or list,
               if `horizon` is an int, we will sample `horizon` step
               continuously after the forecasting point.
               if `horizon` is a list, we will sample discretely according
               to the input list. 1 means the timestamp just after the observed data.
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
        if id_sensitive and not _check_is_aligned(self.df, self.id_col, self.dt_col):
            raise AssertionError("The time series data should be "
                                 "aligned if id_sensitive is set to True.")
        feature_col = _to_list(feature_col, "feature_col") if feature_col is not None \
            else self.feature_col
        target_col = _to_list(target_col, "target_col") if target_col is not None \
            else self.target_col
        if self.roll_additional_feature:
            additional_feature_col =\
                list(set(feature_col).intersection(set(self.roll_additional_feature)))
            feature_col =\
                list(set(feature_col) - set(self.roll_additional_feature))
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

        if lookback == 'auto':
            lookback = self.get_cycle_length('mode', top_k=3)
        rolling_result = \
            self.df.groupby([self.id_col]) \
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
                                      axis=concat_axis).astype(np.float32)
        if horizon != 0:
            self.numpy_y = np.concatenate([rolling_result[i][1]
                                           for i in self._id_list],
                                          axis=concat_axis).astype(np.float32)
        else:
            self.numpy_y = None

        # target first
        if self.id_sensitive:
            feature_start_idx = num_target_col * num_id
            reindex_list = [list(range(i * num_target_col, (i + 1) * num_target_col)) +
                            list(range(feature_start_idx + i * num_feature_col,
                                       feature_start_idx + (i + 1) * num_feature_col))
                            for i in range(num_id)]
            reindex_list = functools.reduce(lambda a, b: a + b, reindex_list)
            sorted_index = sorted(range(len(reindex_list)), key=reindex_list.__getitem__)
            self.numpy_x = self.numpy_x[:, :, sorted_index]

        # scaler index
        num_roll_target = len(self.roll_target)
        repeat_factor = len(self._id_list) if self.id_sensitive else 1
        scaler_index = [self.target_col.index(self.roll_target[i])
                        for i in range(num_roll_target)] * repeat_factor
        self.scaler_index = scaler_index

        return self

    def to_torch_data_loader(self,
                             batch_size=32,
                             roll=False,
                             lookback='auto',
                             horizon=None,
                             feature_col=None,
                             target_col=None, ):
        """
        Convert TSDataset to a PyTorch DataLoader with or without rolling. We recommend to use
        to_torch_data_loader(roll=True) if you don't need to output the rolled numpy array. It is
        much more efficient than rolling separately, especially when the dataframe or lookback
        is large.

        :param batch_size: int, the batch_size for a Pytorch DataLoader. It defaults to 32.
        :param roll: Boolean. Whether to roll the dataframe before converting to DataLoader.
               If True, you must also specify lookback and horizon for rolling. If False, you must
               have called tsdataset.roll() before calling to_torch_data_loader(). Default to False.
        :param lookback: int, lookback value. Default to 'auto',
               the mode of time series' cycle length will be taken as the lookback.
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

        :return: A pytorch DataLoader instance.

        to_torch_data_loader() can be called by:

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
        >>> data_loader = tsdataset.to_torch_data_loader(batch_size=32,
        >>>                                              roll=True,
        >>>                                              lookback=lookback,
        >>>                                              horizon=horizon)
        >>> # or roll outside. That might be less efficient than the way above.
        >>> tsdataset.roll(lookback=lookback, horizon=horizon, id_sensitive=False)
        >>> x, y = tsdataset.to_numpy()
        >>> print(x, y) # x = [[[1.9, 1, 2 ]], [[2.3, 0, 9 ]]] y = [[[ 2.4 ]], [[ 2.6 ]]]
        >>> data_loader = tsdataset.to_torch_data_loader(batch_size=32)

        """
        from torch.utils.data import TensorDataset, DataLoader
        import torch
        if roll:
            if horizon is None:
                raise ValueError("You must input horizon if roll is True")
            from bigdl.chronos.data.utils.roll_dataset import RollDataset
            feature_col = _to_list(feature_col, "feature_col") if feature_col is not None \
                else self.feature_col
            target_col = _to_list(target_col, "target_col") if target_col is not None \
                else self.target_col

            # set scaler index for unscale_numpy
            self.scaler_index = [self.target_col.index(t) for t in target_col]

            if lookback == 'auto':
                lookback = self.get_cycle_length('mode', top_k=3)
            torch_dataset = RollDataset(self.df,
                                        lookback=lookback,
                                        horizon=horizon,
                                        feature_col=feature_col,
                                        target_col=target_col,
                                        id_col=self.id_col)
            return DataLoader(torch_dataset,
                              batch_size=batch_size,
                              shuffle=True)
        else:
            if self.numpy_x is None:
                raise RuntimeError("Please call 'roll' method before transforming a TSDataset to "
                                   "torch DataLoader without rolling (default roll=False)!")
            x, y = self.to_numpy()
            return DataLoader(TensorDataset(torch.from_numpy(x).float(),
                                            torch.from_numpy(y).float()),
                              batch_size=batch_size,
                              shuffle=True)

    def to_numpy(self):
        '''
        Export rolling result in form of a tuple of numpy ndarray (x, y).

        :return: a 2-dim tuple. each item is a 3d numpy ndarray. The ndarray
                 is casted to float32.
        '''
        if self.numpy_x is None:
            raise RuntimeError("Please call 'roll' method "
                               "before transform a TSDataset to numpy ndarray!")
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
        if self.roll_additional_feature:
            feature_col = []
            for feature in self.feature_col:
                if feature not in self.roll_additional_feature:
                    feature_col.append(feature)
        if fit:
            self.df[self.target_col + feature_col] = \
                scaler.fit_transform(self.df[self.target_col + feature_col])
        else:
            from sklearn.utils.validation import check_is_fitted
            try:
                assert not check_is_fitted(scaler)
            except Exception:
                raise AssertionError("When calling scale for the first time, "
                                     "you need to set fit=True.")
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
        if self.roll_additional_feature:
            feature_col = []
            for feature in self.feature_col:
                if feature not in self.roll_additional_feature:
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

    def _check_basic_invariants(self, strict_check=False):
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
            if self.roll_additional_feature and feature_col_name in self.roll_additional_feature:
                continue
            _check_col_within(self.df, feature_col_name)

        # check no n/a in critical col
        _check_col_no_na(self.df, self.dt_col)
        _check_col_no_na(self.df, self.id_col)

        # check dt sorted
        if strict_check:
            _check_dt_is_sorted(self.df, self.dt_col)

    def get_cycle_length(self, aggregate='mode', top_k=3):
        """
        Calculate the cycle length of the time series in this TSDataset.

        Args:
            top_k (int): The freq with top top_k power after fft will be
                used to check the autocorrelation. Higher top_k might be time-consuming.
                The value is default to 3.
            aggregate (str): Select the mode of calculation time period,
                We only support 'min', 'max', 'mode', 'median', 'mean'.

        Returns:
            Describe the value of the time period distribution.
        """
        assert isinstance(top_k, int),\
            f"top_k type must be int, but found {type(top_k)}."
        assert isinstance(aggregate, str),\
            f"aggregate type must be str, but found {type(aggregate)}."
        assert aggregate.lower().strip() in ['min', 'max', 'mode', 'median', 'mean'], \
            f"We Only support 'min' 'max' 'mode' 'median' 'mean', but found {aggregate}."

        if len(self.target_col) == 1:
            res = self.df.groupby(self.id_col)\
                         .apply(lambda x: (cycle_length_est(x[self.target_col[0]].values, top_k)))
        else:
            res = self.df.groupby(self.id_col)\
                         .apply(lambda x: pd.DataFrame({'cycle_length':
                                [cycle_length_est(x[col].values,
                                                  top_k)for col in self.target_col]}))
            res = res.cycle_length

        if aggregate.lower().strip() == 'mode':
            self.best_cycle_length = int(res.value_counts().index[0])
        elif aggregate.lower().strip() == 'mean':
            self.best_cycle_length = int(res.mean())
        elif aggregate.lower().strip() == 'median':
            self.best_cycle_length = int(res.median())
        elif aggregate.lower().strip() == 'min':
            self.best_cycle_length = int(res.min())
        elif aggregate.lower().strip() == 'max':
            self.best_cycle_length = int(res.max())

        return self.best_cycle_length
