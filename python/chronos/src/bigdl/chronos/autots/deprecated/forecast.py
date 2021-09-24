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

from zoo.chronos.autots.deprecated.regression.time_sequence_predictor import TimeSequencePredictor
from zoo.chronos.autots.deprecated.config.recipe import *
from zoo.chronos.autots.deprecated.pipeline.time_sequence import load_ts_pipeline
from zoo.chronos.utils import deprecated


@deprecated('Please use `zoo.chronos.autots.AutoTSEstimator` instead.')
class AutoTSTrainer:
    """
    The Automated Time Series Forecast Trainer
    """
    def __init__(self,
                 horizon=1,
                 dt_col="datetime",
                 target_col="value",
                 logs_dir="~/zoo_automl_logs",
                 extra_features_col=None,
                 search_alg=None,
                 search_alg_params=None,
                 scheduler=None,
                 scheduler_params=None,
                 name="automl"
                 ):
        """
        __init__()
        Initialize the AutoTS Trainer.

        :param horizon: steps to look forward
        :param dt_col: the datetime column
        :param target_col: the target column to forecast
        :param extra_features_col: extra feature columns
        """
        target_col_list = target_col
        if isinstance(target_col, str):
            target_col_list = [target_col]
        self.internal = TimeSequencePredictor(
            dt_col=dt_col,
            target_col=target_col_list,
            logs_dir=logs_dir,
            future_seq_len=horizon,
            extra_features_col=extra_features_col,
            search_alg=search_alg,
            search_alg_params=search_alg_params,
            scheduler=scheduler,
            scheduler_params=scheduler_params,
            name=name
        )

    def fit(self,
            train_df,
            validation_df=None,
            metric="mse",
            recipe: Recipe = SmokeRecipe(),
            uncertainty: bool = False,
            upload_dir=None,
            ):
        """
        fit()
        Fit a time series forecasting pipeline w/ automl

        :param train_df: the input dataframe (as pandas.dataframe)
        :param validation_df: the validation dataframe (as pandas.dataframe)
        :param recipe: the configuration of searching
        :param metric: the evaluation metric to optimize
        :param uncertainty: whether to enable uncertainty calculation
                            (will output an uncertainty sigma)
        :param upload_dir: Optional URI to sync training results and checkpoints. We only support
            hdfs URI for now.
        :return a TSPipeline
        """
        zoo_pipeline = self.internal.fit(train_df,
                                         validation_df,
                                         metric,
                                         recipe,
                                         mc=uncertainty,
                                         upload_dir=upload_dir)
        ppl = TSPipeline()
        ppl.internal = zoo_pipeline
        return ppl


class TSPipeline:
    """
    A pipeline for time series forecasting.
    """

    def __init__(self):
        """
        Initialize an emtpy TSPipeline.
        Usually it is not called by user directly.
        A TSPipeline is either obtained from AutoTrainer.fit or TSPipeline.load
        """
        self.internal = None
        self.uncertainty = False

    def save(self, pipeline_file):
        """
        Save the pipeline to a file

        :param pipeline_file: the file path
        :return:
        """
        return self.internal.save(pipeline_file)

    @staticmethod
    @deprecated('Please use `zoo.chronos.autots.TSPipeline` instead.')
    def load(pipeline_file):
        """
        load(pipeline_file)
        Load pipeline from a file

        :param pipeline_file: the pipeline file
        :return: a TSPipeline object
        """
        tsppl = TSPipeline()
        tsppl.internal = load_ts_pipeline(pipeline_file)
        return tsppl

    def fit(self,
            input_df,
            validation_df=None,
            uncertainty: bool = False,
            epochs=1,
            **user_config):
        """
        Incremental Fitting

        :param input_df: the input dataframe
        :param validation_df: the validation dataframe
        :param uncertainty: whether to calculate uncertainty
        :param epochs: number of epochs to train
        :param user_config: user configurations
        :return:
        """
        # TODO refactor automl.Pipeline fit methods to merge the two
        # maybe use another method to apply configs.
        # distinguish between incremental and fit from scratch
        self.uncertainty = uncertainty
        if user_config:
            self.internal.fit_with_fixed_configs(input_df=input_df,
                                                 validation_df=validation_df,
                                                 mc=uncertainty,
                                                 epoch_num=epochs,
                                                 **user_config)
        else:
            self.internal.fit(input_df=input_df,
                              validation_df=validation_df,
                              mc=uncertainty,
                              epoch_num=epochs)

    def predict(self, input_df):
        """
        Prediction.

        :param input_df: the input dataframe
        :return: the forecast results
        """
        if self.uncertainty is True:
            return self.internal.predict_with_uncertainty(input_df)
        else:
            return self.internal.predict(input_df)

    def evaluate(self,
                 input_df,
                 metrics=["mse"],
                 multioutput='raw_values'):
        """
        Evaluation

        :param input_df: the input dataframe
        :param metrics: the evaluation metrics
        :param multioutput: output mode of multiple output, whether to aggregate
        :return: the evaluation results
        """
        return self.internal.evaluate(input_df, metrics, multioutput)
