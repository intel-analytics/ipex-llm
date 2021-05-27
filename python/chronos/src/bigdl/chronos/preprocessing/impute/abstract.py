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

import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from abc import ABC, abstractmethod


# migrate from automl/input
class BaseImputation(ABC):
    """
    Abstract Base class for imputation.
    """

    @abstractmethod
    def impute(self, input_df):
        """
        fill na value
        :param input_df: input dataframe
        :return:
        """
        pass

    def save(self, file_path):
        """
        save the feature tools internal variables.
        Some of the variables are derived after fit_transform, so only saving config is not enough.
        :param: file_path : the file to be saved
        :param: config: the trial config
        :return:
        """
        pass

    def restore(self, **config):
        """
        Restore variables from file
        :param file_path: file contain saved parameters.
                          i.e. some parameters are obtained during training,
                          not in trial config, e.g. scaler fit params)
        :param config: the trial config
        :return:
        """
        pass


class BaseImpute(ABC):
    """
    base model for data imputation
    """

    @abstractmethod
    def impute(self, df):
        """
        fill in missing values in the dataframe
        :param df: dataframe containing missing values
        :return: dataframe without missing values
        """
        raise NotImplementError

    def evaluate(self, df, drop_rate):
        """
        evaluate model by randomly drop some value
        :params df: input dataframe
        :params drop_rate: percentage value that will be randomly dropped
        :return: MSE results
        """
        missing = df.isna()*1
        df1 = self.impute(df)
        missing = missing.to_numpy()
        mask = np.zeros(df.shape[0]*df.shape[1])
        idx = np.random.choice(mask.shape[0], int(mask.shape[0]*drop_rate), replace=False)
        mask[idx] = 1
        mask = np.reshape(mask, (df.shape[0], df.shape[1]))
        np_df = df.to_numpy()
        np_df[mask == 1] = None
        new_df = pd.DataFrame(np_df)
        impute_df = self.impute(new_df)
        pred = impute_df.to_numpy()
        true = df1.to_numpy()
        pred[missing == 1] = 0
        true[missing == 1] = 0
        result = []
        for i in range(len(df.columns)):
            result.append(metrics.mean_squared_error(true[:, i], pred[:, i]))
        return result
