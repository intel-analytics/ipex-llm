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

from zoo.chronos.autots.deprecated.feature.utils import save_config
from zoo.chronos.autots.deprecated.feature.abstract import BaseFeatureTransformer


class IdentityTransformer(BaseFeatureTransformer):
    """
    echo transformer
    """
    def __init__(self,
                 feature_cols=None,
                 target_col=None):
        self.feature_cols = feature_cols
        self.target_col = target_col

    def fit_transform(self, input_df, **config):
        train_x = input_df[self.feature_cols]
        train_y = input_df[[self.target_col]]
        return train_x, train_y

    def transform(self, input_df, is_train=True):
        train_x = input_df[self.feature_cols]
        train_y = input_df[[self.target_col]]
        return train_x, train_y

    def save(self, file_path, replace=False):
        data_to_save = {"feature_cols": self.feature_cols,
                        "target_col": self.target_col
                        }
        save_config(file_path, data_to_save, replace=replace)

    def restore(self, **config):
        self.feature_cols = config["feature_cols"]
        self.target_col = config["target_col"]

    def _get_required_parameters(self):
        return set()

    def _get_optional_parameters(self):
        return set()

    def post_processing(self, input_df, y_pred, is_train):
        if is_train:
            return input_df[[self.target_col]], y_pred
        else:
            return y_pred
