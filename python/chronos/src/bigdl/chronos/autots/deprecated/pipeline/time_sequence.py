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
import os
import time

from bigdl.chronos.autots.deprecated.feature.utils import save_config
from bigdl.chronos.autots.deprecated.pipeline.base import Pipeline
from bigdl.chronos.autots.deprecated.model.time_sequence import TimeSequenceModel
from bigdl.chronos.autots.deprecated.pipeline.parameters import DEFAULT_CONFIG_DIR, DEFAULT_PPL_DIR
from bigdl.chronos.utils import deprecated


class TimeSequencePipeline(Pipeline):

    def __init__(self, model=None, name=None):
        """
        initialize a pipeline
        :param model: the internal model
        """
        self.model = model
        self.config = self.model.config
        self.name = name
        self.time = time.strftime("%Y%m%d-%H%M%S")

    def describe(self):
        init_info = ['future_seq_len', 'dt_col', 'target_col', 'extra_features_col', 'drop_missing']
        print("**** Initialization info ****")
        for info in init_info:
            print(info + ":", getattr(self.model.ft, info))
        print("")

    def fit(self, input_df, validation_df=None, mc=False, epoch_num=20):
        self.model.fit_incr(input_df, validation_df, mc=mc, verbose=1, epochs=epoch_num)
        print('Fit done!')

    def fit_with_fixed_configs(self, input_df, validation_df=None, mc=False, **user_configs):
        """
        Fit pipeline with fixed configs. The model will be trained from initialization
        with the hyper-parameter specified in configs. The configs contain both identity configs
        (Eg. "future_seq_len", "dt_col", "target_col", "metric") and automl tunable configs
        (Eg. "past_seq_len", "batch_size").
        We recommend calling get_default_configs to see the name and default values of configs you
        you can specify.
        :param input_df: one data frame or a list of data frames
        :param validation_df: one data frame or a list of data frames
        :param user_configs: you can overwrite or add more configs with user_configs. Eg. "epochs"
        :return:
        """
        # self._check_configs()
        config = self.config.copy()
        config.update(user_configs)

        self.model.setup(config)
        self.model.fit_eval(data=input_df,
                            validation_data=validation_df,
                            mc=mc,
                            verbose=1, **config)

    def evaluate(self,
                 input_df,
                 metrics=["mse"],
                 multioutput='raw_values'
                 ):
        """
        evaluate the pipeline
        :param input_df:
        :param metrics: subset of ['mean_squared_error', 'r_square', 'sMAPE']
        :param multioutput: string in ['raw_values', 'uniform_average']
                'raw_values' :
                    Returns a full set of errors in case of multioutput input.
                'uniform_average' :
                    Errors of all outputs are averaged with uniform weight.
        :return:
        """
        return self.model.evaluate(df=input_df, metric=metrics)

    def predict(self, input_df):
        """
        predict test data with the pipeline fitted
        :param input_df:
        :return:
        """
        return self.model.predict(df=input_df)

    def predict_with_uncertainty(self, input_df, n_iter=100):
        return self.model.predict_with_uncertainty(input_df, n_iter)

    def save(self, ppl_file=None):
        """
        save pipeline to file, contains feature transformer, model, trial config.
        :param ppl_file:
        :return:
        """
        ppl_file = ppl_file or os.path.join(DEFAULT_PPL_DIR, "{}_{}.ppl".
                                            format(self.name, self.time))
        self.model.save(ppl_file)
        return ppl_file

    def config_save(self, config_file=None):
        """
        save all configs to file.
        :param config_file:
        :return:
        """
        config_file = config_file or os.path.join(DEFAULT_CONFIG_DIR, "{}_{}.json".
                                                  format(self.name, self.time))
        save_config(config_file, self.config, replace=True)
        return config_file


@deprecated('Please use `bigdl.chronos.autots.TSPipeline` instead.')
def load_ts_pipeline(file):
    model = TimeSequenceModel()
    model.restore(file)
    print("Restore pipeline from", file)
    return TimeSequencePipeline(model)
