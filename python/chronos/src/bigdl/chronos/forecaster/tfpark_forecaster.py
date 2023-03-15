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

from abc import ABCMeta, abstractmethod
from bigdl.orca.tfpark import KerasModel as TFParkKerasModel
import tensorflow as tf
from bigdl.chronos.forecaster.abstract import Forecaster


class TFParkForecaster(TFParkKerasModel, Forecaster, metaclass=ABCMeta):
    """
    Base class for TFPark KerasModel based Forecast models.
    """

    def __init__(self):
        """
        Build a tf.keras model.
        Turns the tf.keras model returned from _build into a tfpark.KerasModel
        """
        self.model = self._build()
        from bigdl.nano.utils.common import invalidInputError
        invalidInputError((isinstance(self.model, tf.keras.Model),
                           "expect model is tf.keras.Model"))
        super().__init__(self.model)

    @abstractmethod
    def _build(self):
        """
        Build a tf.keras model.

        :return: a tf.keras model (compiled)
        """
        pass
