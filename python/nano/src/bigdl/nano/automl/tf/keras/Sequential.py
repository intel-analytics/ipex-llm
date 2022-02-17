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

from bigdl.nano.automl.utils.lazyutils import proxy_methods
from bigdl.nano.automl.hpo.mixin import HPOMixin
from bigdl.nano.automl.hpo.backend import OptunaBackend
from bigdl.nano.automl.hpo.space import AutoGluonObject


@proxy_methods
class Sequential(HPOMixin, tf.keras.Sequential):


    def __init__(self, layers=None, name=None):
        super().__init__(layers=None, name=name)
        # TODO add more flexibility for args parsing
        #self.init_args = args
        #self.init_kwargs = kwargs
        self.name_ = name
        self.lazylayers_ = layers if layers is not None else []

        self.objective = None
        self.study = None
        self.tune_end=False
        self._lazymodel = None

    def add(self, layer):
        # just add all layers into a cache
        # and layer will be instantiated later
        self.lazylayers_.append(layer)

    def _model_init_args(self, trial):
        # for lazy model init
        # use backend to sample model init args
        # and construct the actual layers
        instantiated_layers=[]
        for l in self.lazylayers_:
            if isinstance(l, AutoGluonObject):
                layer = OptunaBackend.instantiate(trial, l)
                instantiated_layers.append(layer)
            else:
                newl = copy.deepcopy(l)
                instantiated_layers.append(newl)
        return {'layers':instantiated_layers, 'name': self.name_}





