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
from bigdl.orca.automl.xgboost.XGBoost import XGBoostModelBuilder
from bigdl.orca.automl.auto_estimator import AutoEstimator


class AutoXGBClassifier(AutoEstimator):
    def __init__(self,
                 logs_dir="/tmp/auto_xgb_classifier_logs",
                 cpus_per_trial=1,
                 name=None,
                 **xgb_configs
                 ):
        """
        Automated xgboost classifier

        Example:
            >>> search_space = {"n_estimators": hp.grid_search([50, 1000]),
                                "max_depth": hp.grid_search([2, 15]),
                                "lr": hp.loguniform(1e-4, 1e-1)}
            >>> auto_xgb_clf = AutoXGBClassifier(cpus_per_trial=4,
                                                 name="auto_xgb_classifier",
                                                 **config)
            >>> auto_xgb_clf.fit(data=(X_train, y_train),
                                 validation_data=(X_val, y_val),
                                 metric="error",
                                 metric_mode="min",
                                 n_sampling=1,
                                 search_space=search_space)
            >>> best_model = auto_xgb_clf.get_best_model()

        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_xgb_classifier_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. It defaults to 1.
            The value will also be assigned to n_jobs in xgboost,
            which is the number of parallel threads used to run xgboost.
        :param name: Name of the auto xgboost classifier.
        :param xgb_configs: Other scikit learn xgboost parameters. You may refer to
           https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
           for the parameter names to specify. Note that we will directly use cpus_per_trial value
           for n_jobs in xgboost and you shouldn't specify n_jobs again.
        """
        xgb_model_builder = XGBoostModelBuilder(model_type='classifier',
                                                cpus_per_trial=cpus_per_trial,
                                                **xgb_configs)
        resources_per_trial = {"cpu": cpus_per_trial} if cpus_per_trial else None
        super().__init__(model_builder=xgb_model_builder,
                         logs_dir=logs_dir,
                         resources_per_trial=resources_per_trial,
                         name=name)


class AutoXGBRegressor(AutoEstimator):
    def __init__(self,
                 logs_dir="/tmp/auto_xgb_regressor_logs",
                 cpus_per_trial=1,
                 name=None,
                 **xgb_configs
                 ):
        """
        Automated xgboost regressor

        Example:
            >>> search_space = {"n_estimators": hp.grid_search([800, 1000]),
                                "max_depth": hp.grid_search([10, 15]),
                                "lr": hp.loguniform(1e-4, 1e-1),
                                "min_child_weight": hp.choice([1, 2, 3]),
                                }
            >>> auto_xgb_reg = AutoXGBRegressor(cpus_per_trial=2,
                                                name="auto_xgb_regressor",
                                                **config)
            >>> auto_xgb_reg.fit(data=(X_train, y_train),
                                 validation_data=(X_val, y_val),
                                 metric="rmse",
                                 n_sampling=1,
                                 search_space=search_space)
            >>> best_model = auto_xgb_reg.get_best_model()

        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_xgb_classifier_logs"
        :param cpus_per_trial: Int. Number of cpus for each trial. The value will also be assigned
            to n_jobs, which is the number of parallel threads used to run xgboost.
        :param name: Name of the auto xgboost classifier.
        :param xgb_configs: Other scikit learn xgboost parameters. You may refer to
           https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
           for the parameter names to specify. Note that we will directly use cpus_per_trial value
           for n_jobs in xgboost and you shouldn't specify n_jobs again.
        """
        xgb_model_builder = XGBoostModelBuilder(model_type='regressor',
                                                cpus_per_trial=cpus_per_trial,
                                                **xgb_configs)
        resources_per_trial = {"cpu": cpus_per_trial} if cpus_per_trial else None
        super().__init__(model_builder=xgb_model_builder,
                         logs_dir=logs_dir,
                         resources_per_trial=resources_per_trial,
                         name=name)
