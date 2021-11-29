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

from bigdl.orca.test_zoo_utils import ZooTestCase

import numpy as np
import pandas as pd
import os
from numpy.testing import assert_array_almost_equal

from bigdl.orca.automl.xgboost.XGBoost import XGBoost, XGBoostModelBuilder
import pytest


class TestXgbregressor(ZooTestCase):
    def setup_method(self, method):
        # super().setup_method(method)
        self.model = XGBoost(config={'n_estimators': 5, 'max_depth': 2, 'tree_method': 'hist'})
        feature_cols = ["f", "f2"]
        target_col = "t"
        train_df = pd.DataFrame({"f": np.random.randn(20),
                                 "f2": np.random.randn(20),
                                 "t": np.random.randint(20)})
        val_df = pd.DataFrame({"f": np.random.randn(5),
                               "f2": np.random.randn(5),
                               "t": np.random.randint(5)})

        self.x, self.y = train_df[feature_cols], train_df[[target_col]]
        self.val_x, self.val_y = val_df[feature_cols], val_df[[target_col]]

    def teardown_method(self, method):
        pass

    def test_fit_predict_evaluate(self):
        self.model.fit_eval((self.x, self.y), [(self.val_x, self.val_y)])

        # test predict
        result = self.model.predict(self.val_x)

        # test evaluate
        evaluate_result = self.model.evaluate(self.val_x, self.val_y)

    def test_save_restore(self):
        self.model.fit_eval((self.x, self.y), [(self.val_x, self.val_y)])

        result_save = self.model.predict(self.val_x)
        model_file = "tmp.pkl"
        self.model.save(model_file)
        assert os.path.isfile(model_file)
        new_model = XGBoost()
        new_model.restore(model_file)
        assert new_model.model
        result_restore = new_model.predict(self.val_x)
        assert_array_almost_equal(result_save, result_restore, decimal=2), \
            "Prediction values are not the same after restore: " \
            "predict before is {}, and predict after is {}".format(result_save, result_restore)
        os.remove(model_file)

    def test_metric(self):
        # metric not in XGB_METRIC_NAME but in Evaluator.metrics_func.keys()
        self.model.fit_eval(
            data=(self.x, self.y),
            validation_data=[(self.val_x, self.val_y)],
            metric="mse")
        # metric in XGB_METRIC_NAME
        self.model.fit_eval(
            data=(self.x, self.y),
            validation_data=[(self.val_x, self.val_y)],
            metric="rmsle")

        with pytest.raises(ValueError):
            self.model.fit_eval(
                data=(self.x, self.y),
                validation_data=[(self.val_x, self.val_y)],
                metric="wrong_metric")

        # metric func
        def pyrmsle(y_true, y_pred):
            y_pred[y_pred < -1] = -1 + 1e-6
            elements = np.power(np.log1p(y_true) - np.log1p(y_pred), 2)
            return float(np.sqrt(np.sum(elements) / len(y_true)))

        result = self.model.fit_eval(
            data=(self.x, self.y),
            validation_data=[(self.val_x, self.val_y)],
            metric_func=pyrmsle)
        assert "pyrmsle" in result

    def test_data_creator(self):
        def get_x_y(size, config):
            values = np.random.randn(size, 4)
            df = pd.DataFrame(values, columns=["f1", "f2", "f3", "t"])
            selected_features = config["features"]
            x = df[selected_features].to_numpy()
            y = df["t"].to_numpy()
            return x, y

        from functools import partial
        train_data_creator = partial(get_x_y, 20)
        val_data_creator = partial(get_x_y, 5)
        config = {'n_estimators': 5, 'max_depth': 2, 'tree_method': 'hist'}
        model_builder = XGBoostModelBuilder(model_type="regressor", cpus_per_trial=1, **config)
        model = model_builder.build(config={"features": ["f1", "f2"]})
        model.fit_eval(train_data_creator, validation_data=val_data_creator, metric="mae")


if __name__ == "__main__":
    pytest.main([__file__])
