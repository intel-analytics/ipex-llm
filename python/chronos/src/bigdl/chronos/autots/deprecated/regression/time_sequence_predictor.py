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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp'
# ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from bigdl.chronos.autots.deprecated.regression.base_predictor import BasePredictor
from bigdl.chronos.utils import deprecated


@deprecated('Please use `bigdl.chronos.autots.AutoTSEstimator` instead.')
class TimeSequencePredictor(BasePredictor):
    """
    Trains a model that predicts future time sequence from past sequence.
    Past sequence should be > 1. Future sequence can be > 1.
    For example, predict the next 2 data points from past 5 data points.
    Output have only one target value (a scalar) for each data point in the sequence.
    Input can have more than one features (value plus several features)
    Example usage:
        tsp = TimeSequencePredictor()
        tsp.fit(input_df)
        result = tsp.predict(test_df)

    """
    def __init__(self,
                 name="automl",
                 logs_dir="~/bigdl_automl_logs",
                 future_seq_len=1,
                 dt_col="datetime",
                 target_col=["value"],
                 extra_features_col=None,
                 drop_missing=True,
                 search_alg=None,
                 search_alg_params=None,
                 scheduler=None,
                 scheduler_params=None,
                 ):
        self.pipeline = None
        self.future_seq_len = future_seq_len
        self.dt_col = dt_col
        if isinstance(target_col, str):
            self.target_col = [target_col]
        else:
            self.target_col = target_col
        self.extra_features_col = extra_features_col
        self.drop_missing = drop_missing
        super().__init__(name=name,
                         logs_dir=logs_dir,
                         search_alg=search_alg,
                         search_alg_params=search_alg_params,
                         scheduler=scheduler,
                         scheduler_params=scheduler_params)

    def get_model_builder(self):
        from bigdl.chronos.autots.deprecated.model.time_sequence import TSModelBuilder
        model_builder = TSModelBuilder(
            dt_col=self.dt_col,
            target_cols=self.target_col,
            future_seq_len=self.future_seq_len,
            extra_features_col=self.extra_features_col,
            drop_missing=self.drop_missing,
        )
        return model_builder

    def _check_missing_col(self, df):
        cols_list = [self.dt_col] + self.target_col
        if self.extra_features_col is not None:
            if not isinstance(self.extra_features_col, (list,)):
                raise ValueError(
                    "extra_features_col needs to be either None or a list")
            cols_list.extend(self.extra_features_col)

        missing_cols = set(cols_list) - set(df.columns)
        if len(missing_cols) != 0:
            raise ValueError("Missing Columns in the input data frame:" +
                             ','.join(list(missing_cols)))

    def _check_df(self, df):
        super()._check_df(df)
        self._check_missing_col(df=df)
