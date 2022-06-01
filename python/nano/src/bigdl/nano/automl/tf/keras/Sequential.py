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

import tensorflow as tf
import copy

from bigdl.nano.automl.utils import proxy_methods
from bigdl.nano.automl.tf.mixin import HPOMixin
from bigdl.nano.automl.hpo.space import AutoObject


@proxy_methods
class Sequential(HPOMixin, tf.keras.Sequential):
    """Tf.keras.Sequential with HPO capabilities."""

    def __init__(self, layers=None, name=None):
        """
        Initialzier.

        :param layers: a list of layers (optional). Defults to None.
        :param name: str(optional), name of the model. Defaults to None
        """
        super().__init__(layers=None, name=name)
        # TODO add more flexibility for args parsing
        # self.init_args = args
        # self.init_kwargs = kwargs
        self.model_class = tf.keras.Sequential
        self.name_ = name
        self.lazylayers_ = layers if layers is not None else []

    def add(self, layer):
        """
        Add a layer.

        :param layer: the layer to be added.
        """
        # just add all layers into a cache
        # and layer will be instantiated later
        self.lazylayers_.append(layer)

    def _model_init_args(self, trial):
        # for lazy model init
        # use backend to sample model init args
        # and construct the actual layers
        instantiated_layers = []
        for layer in self.lazylayers_:
            if isinstance(layer, AutoObject):
                newl = self.backend.instantiate(trial, layer)
            else:
                newl = copy.deepcopy(layer)
            instantiated_layers.append(newl)
        return {'layers': instantiated_layers, 'name': self.name_}

    def _get_model_init_args_func_kwargs(self):
        """Return the kwargs of _model_init_args_func except trial."""
        return {
            'lazylayers': self.lazylayers_,
            'name': self.name_,
            'backend': self.backend
        }

    @staticmethod
    def _model_init_args_func(trial, lazylayers, name, backend):
        instantiated_layers = []
        for layer in lazylayers:
            if isinstance(layer, AutoObject):
                newl = backend.instantiate(trial, layer)
            else:
                newl = copy.deepcopy(layer)
            instantiated_layers.append(newl)
        return {'layers': instantiated_layers, 'name': name}
