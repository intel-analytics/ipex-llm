# +
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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either exp'
# ress or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from zoo.orca.automl.auto_estimator import AutoEstimator
from zoo.chronos.model.prophet import ProphetBuilder


# -

class AutoProphet(AutoEstimator):

    def __init__(self,
                 logs_dir="/tmp/auto_prophet_logs",
                 **prophet_config
                 ):
        """
        Automated Prophet Model
        :param logs_dir: Local directory to save logs and results. It defaults to
            "/tmp/auto_prophet_logs"
        :param prophet_config: Other prophet hyperparameters. You may refer to
           https://facebook.github.io/prophet/docs/diagnostics.html#hyperparameter-tuning
        for the parameter names to specify.
        """
        prophet_model_builder = ProphetBuilder(**prophet_config)
        super().__init__(model_builder=prophet_model_builder,
                         logs_dir=logs_dir,)
