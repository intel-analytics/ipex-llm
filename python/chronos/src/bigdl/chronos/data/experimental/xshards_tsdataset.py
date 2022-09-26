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


from bigdl.orca.data.shard import SparkXShards
from bigdl.orca.learn.utils import dataframe_to_xshards_of_pandas_df
from bigdl.chronos.data.utils.utils import _to_list, _check_type
from bigdl.chronos.data.utils.roll import roll_timeseries_dataframe
from bigdl.chronos.data.utils.impute import impute_timeseries_dataframe
from bigdl.chronos.data.utils.split import split_timeseries_dataframe
from bigdl.chronos.data.experimental.utils import add_row, transform_to_dict
from bigdl.chronos.data.utils.scale import unscale_timeseries_numpy


_DEFAULT_ID_COL_NAME = "id"
_DEFAULT_ID_PLACEHOLDER = "0"


class XShardsTSDataset:

    def __init__(self, shards, **schema):
        '''
        XShardTSDataset is an abstract of time series dataset with distributed fashion.
        Cascade call is supported for most of the transform methods.
        XShardTSDataset will partition the dataset by id_col, which is experimental.
        '''
        self.shards = shards
        self.id_col = schema["id_col"]
        self.dt_col = schema["dt_col"]
        self.feature_col = schema["feature_col"].copy()
        self.target_col = schema["target_col"].copy()
        self.scaler_index = [i for i in range(len(self.target_col))]

        self.numpy_shards = None

        self._id_list = list(shards[self.id_col].unique())

    @staticmethod
    def from_xshards(shards,
                     dt_col,
                     target_col,
                     id_col=None,
                     extra_feature_col=None,
                     with_split=False,
                     val_ratio=0,
                     test_ratio=0.1):
        '''
        Initialize xshardtsdataset(s) from xshard pandas dataframe.

        :param shards: an xshards pandas dataframe for your raw time series data.
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

        :return: a XShardTSDataset instance when with_split is set to False,
                 three XShardTSDataset instances when with_split is set to True.

        Create a xshardtsdataset instance by:

        >>> # Here is a df example:
        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"
        >>> # 00        2019-01-01    1.9     1                   2
        >>> # 01        2019-01-01    2.3     0                   9
        >>> # 00        2019-01-02    2.4     3                   4
        >>> # 01        2019-01-02    2.6     0                   2
        >>> from bigdl.orca.data.pandas import read_csv
        >>> shards = read_csv(csv_path)
        >>> tsdataset = XShardsTSDataset.from_xshards(shards, dt_col="datetime",
        >>>                                           target_col="value", id_col="id",
        >>>                                           extra_feature_col=["extra feature 1",
        >>>                                                              "extra feature 2"])
        '''

        _check_type(shards, "shards", SparkXShards)

        target_col = _to_list(target_col, name="target_col")
        feature_col = _to_list(extra_feature_col, name="extra_feature_col")

        if id_col is None:
            shards = shards.transform_shard(add_row,
                                            _DEFAULT_ID_COL_NAME,
                                            _DEFAULT_ID_PLACEHOLDER)
            id_col = _DEFAULT_ID_COL_NAME

        # repartition to id
        shards = shards.partition_by(cols=id_col,
                                     num_partitions=len(shards[id_col].unique()))

        if with_split:
            tsdataset_shards\
                = shards.transform_shard(split_timeseries_dataframe,
                                         id_col, val_ratio, test_ratio).split()
            return [XShardsTSDataset(shards=tsdataset_shards[i],
                                     id_col=id_col,
                                     dt_col=dt_col,
                                     target_col=target_col,
                                     feature_col=feature_col) for i in range(3)]

        return XShardsTSDataset(shards=shards,
                                id_col=id_col,
                                dt_col=dt_col,
                                target_col=target_col,
                                feature_col=feature_col)

    @staticmethod
    def from_sparkdf(df,
                     dt_col,
                     target_col,
                     id_col=None,
                     extra_feature_col=None,
                     with_split=False,
                     val_ratio=0,
                     test_ratio=0.1):
        '''
        Initialize xshardtsdataset(s) from Spark Dataframe.

        :param df: an Spark DataFrame for your raw time series data.
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

        :return: a XShardTSDataset instance when with_split is set to False,
                 three XShardTSDataset instances when with_split is set to True.

        Create a xshardtsdataset instance by:

        >>> # Here is a df example:
        >>> # id        datetime      value   "extra feature 1"   "extra feature 2"
        >>> # 00        2019-01-01    1.9     1                   2
        >>> # 01        2019-01-01    2.3     0                   9
        >>> # 00        2019-01-02    2.4     3                   4
        >>> # 01        2019-01-02    2.6     0                   2
        >>> df = <pyspark.sql.dataframe.DataFrame>
        >>> tsdataset = XShardsTSDataset.from_xshards(df, dt_col="datetime",
        >>>                                           target_col="value", id_col="id",
        >>>                                           extra_feature_col=["extra feature 1",
        >>>                                                              "extra feature 2"])
        '''

        from pyspark.sql.dataframe import DataFrame
        _check_type(df, "df", DataFrame)

        target_col = _to_list(target_col, name="target_col")
        feature_col = _to_list(extra_feature_col, name="extra_feature_col")
        all_col = target_col + feature_col + _to_list(id_col, name="id_col") + [dt_col]

        shards = dataframe_to_xshards_of_pandas_df(df,
                                                   feature_cols=all_col,
                                                   label_cols=None,
                                                   accept_str_col=False)

        if id_col is None:
            shards = shards.transform_shard(add_row,
                                            _DEFAULT_ID_COL_NAME,
                                            _DEFAULT_ID_PLACEHOLDER)
            id_col = _DEFAULT_ID_COL_NAME

        # repartition to id
        shards = shards.partition_by(cols=id_col,
                                     num_partitions=len(shards[id_col].unique()))

        if with_split:
            tsdataset_shards\
                = shards.transform_shard(split_timeseries_dataframe,
                                         id_col, val_ratio, test_ratio).split()
            return [XShardsTSDataset(shards=tsdataset_shards[i],
                                     id_col=id_col,
                                     dt_col=dt_col,
                                     target_col=target_col,
                                     feature_col=feature_col) for i in range(3)]

        return XShardsTSDataset(shards=shards,
                                id_col=id_col,
                                dt_col=dt_col,
                                target_col=target_col,
                                feature_col=feature_col)

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
               you used to initialize the xshardtsdataset.
        :param id_sensitive: bool,
               |if `id_sensitive` is False, we will rolling on each id's sub dataframe
               |and fuse the sampings.
               |The shape of rolling will be
               |x: (num_sample, lookback, num_feature_col + num_target_col)
               |y: (num_sample, horizon, num_target_col)
               |where num_sample is the summation of sample number of each dataframe
               |if `id_sensitive` is True, we have not implement this currently.

        :return: the xshardtsdataset instance.
        '''
        from bigdl.nano.utils.log4Error import invalidInputError
        if id_sensitive:
            invalidInputError(False,
                              "id_sensitive option has not been implemented.")
        feature_col = _to_list(feature_col, "feature_col") if feature_col is not None \
            else self.feature_col
        target_col = _to_list(target_col, "target_col") if target_col is not None \
            else self.target_col
        self.numpy_shards = self.shards.transform_shard(roll_timeseries_dataframe,
                                                        None, lookback, horizon,
                                                        feature_col, target_col,
                                                        self.id_col, 0, True)
        self.scaler_index = [self.target_col.index(target_col[i])
                             for i in range(len(target_col))]

        return self

    def scale(self, scaler, fit=True):
        '''
        Scale the time series dataset's feature column and target column.

        :param scaler: a dictionary of scaler instance, where keys are id name
               and values are corresponding scaler instance. e.g. if you have
               2 ids called "id1" and "id2", a legal scaler input can be
               {"id1": StandardScaler(), "id2": StandardScaler()}
        :param fit: if we need to fit the scaler. Typically, the value should
               be set to True for training set, while False for validation and
               test set. The value is defaulted to True.

        :return: the xshardtsdataset instance.

        Assume there is a training set tsdata and a test set tsdata_test.
        scale() should be called first on training set with default value fit=True,
        then be called on test set with the same scaler and fit=False.

        >>> from sklearn.preprocessing import StandardScaler
        >>> scaler = {"id1": StandardScaler(), "id2": StandardScaler()}
        >>> tsdata.scale(scaler, fit=True)
        >>> tsdata_test.scale(scaler, fit=False)
        '''
        def _fit(df, id_col, scaler, feature_col, target_col):
            '''
            This function is used to fit scaler dictionary on each shard.
            returns a dictionary of id-scaler pair for each shard.

            Note: this function will not transform the shard.
            '''
            scaler_for_this_id = scaler[df[id_col].iloc[0]]
            df[target_col + feature_col] = scaler_for_this_id.fit(df[target_col + feature_col])

            return {id_col: df[id_col].iloc[0], "scaler": scaler_for_this_id}

        def _transform(df, id_col, scaler, feature_col, target_col):
            '''
            This function is used to transform the shard by fitted scaler.

            Note: this function will not fit the scaler.
            '''
            from sklearn.utils.validation import check_is_fitted
            from bigdl.nano.utils.log4Error import invalidInputError

            scaler_for_this_id = scaler[df[id_col].iloc[0]]
            invalidInputError(not check_is_fitted(scaler_for_this_id),
                              "scaler is not fitted. When calling scale for the first time, "
                              "you need to set fit=True.")
            df[target_col + feature_col] =\
                scaler_for_this_id.transform(df[target_col + feature_col])

            return df

        if fit:
            self.shards_scaler = self.shards.transform_shard(_fit, self.id_col, scaler,
                                                             self.feature_col, self.target_col)
            self.scaler_dict = self.shards_scaler.collect()
            self.scaler_dict = {sc[self.id_col]: sc["scaler"] for sc in self.scaler_dict}
            scaler.update(self.scaler_dict)  # make the change up-to-date outside the tsdata

            self.shards = self.shards.transform_shard(_transform, self.id_col, self.scaler_dict,
                                                      self.feature_col, self.target_col)
        else:
            self.scaler_dict = scaler
            self.shards = self.shards.transform_shard(_transform, self.id_col, self.scaler_dict,
                                                      self.feature_col, self.target_col)
        return self

    def unscale(self):
        '''
        Unscale the time series dataset's feature column and target column.

        :return: the xshardtsdataset instance.
        '''
        def _inverse_transform(df, id_col, scaler, feature_col, target_col):
            from sklearn.utils.validation import check_is_fitted
            from bigdl.nano.utils.log4Error import invalidInputError

            scaler_for_this_id = scaler[df[id_col].iloc[0]]
            invalidInputError(not check_is_fitted(scaler_for_this_id),
                              "scaler is not fitted. When calling scale for the first time, "
                              "you need to set fit=True.")

            df[target_col + feature_col] =\
                scaler_for_this_id.inverse_transform(df[target_col + feature_col])
            return df

        self.shards = self.shards.transform_shard(_inverse_transform, self.id_col,
                                                  self.scaler_dict, self.feature_col,
                                                  self.target_col)
        return self

    def unscale_xshards(self, data, key=None):
        '''
        Unscale the time series forecaster's numpy prediction result/ground truth.

        :param data: xshards same with self.numpy_xshards.
        :param key: str, 'y' or 'prediction', default to 'y'. if no "prediction"
        or "y" return an error and require our users to input a key. if key is
        None, key will be set 'prediction'.

        :return: the unscaled xshardtsdataset instance.
        '''
        from bigdl.nano.utils.log4Error import invalidInputError

        def _inverse_transform(data, scaler, scaler_index, key):
            from sklearn.utils.validation import check_is_fitted
            from bigdl.nano.utils.log4Error import invalidInputError

            id = data['id'][0, 0]
            scaler_for_this_id = scaler[id]
            invalidInputError(not check_is_fitted(scaler_for_this_id),
                              "scaler is not fitted. When calling scale for the first time, "
                              "you need to set fit=True.")

            return unscale_timeseries_numpy(data[key], scaler_for_this_id, scaler_index)

        if key is None:
            key = 'prediction'
        invalidInputError(key in {'y', 'prediction'}, "key is not in {'y', 'prediction'}, "
                          "please input the correct key.")

        return data.transform_shard(_inverse_transform, self.scaler_dict, self.scaler_index, key)

    def impute(self,
               mode="last",
               const_num=0):
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
        def df_reset_index(df):
                df.reset_index(drop=True, inplace=True)
                return df
        self.shards = self.shards.transform_shard(impute_timeseries_dataframe,
                                                  self.dt_col, mode,
                                                  const_num)
        self.shards = self.shards.transform_shard(df_reset_index)
        return self

    def to_xshards(self, partition_num=None):
        '''
        Export rolling result in form of a dict of numpy ndarray {'x': ..., 'y': ..., 'id': ...},
        where value for 'x' and 'y' are 3-dim numpy ndarray and value for 'id' is 2-dim ndarray
        with shape (batch_size, 1)

        :param partition_num: how many partition you would like to split your data.

        :return: a 3-element dict xshard. each value is a 3d numpy ndarray. The ndarray
                 is casted to float32. Default to None which will partition according
                 to id.
        '''
        from bigdl.nano.utils.log4Error import invalidInputError
        if self.numpy_shards is None:
            invalidInputError(False,
                              "Please call 'roll' method "
                              "before transform a XshardsTSDataset to numpy ndarray!")
        if partition_num is None:
            return self.numpy_shards.transform_shard(transform_to_dict)
        else:
            return self.numpy_shards.transform_shard(transform_to_dict).repartition(partition_num)
