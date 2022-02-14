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

from bigdl.nano.automl.hpo.space import AutoGluonObject
import bigdl.nano.tf.keras.Sequential as NanoSequential
from bigdl.nano.automl.hpo.mixin import HPOMixin
from bigdl.nano.automl.hpo.backend import OptunaBackend
import bigdl.nano.automl.hpo as hpo
import copy

class Sequential(HPOMixin, tf.keras.Sequential):
    def __init__(self, layers=None, name=None):
        super().__init__(layers=None, name=name)
        # TODO add more flexibility for args parsing
        #self.init_args = args
        #self.init_kwargs = kwargs
        self.name_ = name
        self.layers_ = layers
        if self.layers_ is None:
            self.layers_ = []

        self.objective = None
        self.study = None
        self.tune_end=False

    def _istantiate_layers(self,trial, layers):
        inst_layers=[]
        for l in layers:
            if isinstance(l, AutoGluonObject):
                config = OptunaBackend.gen_config(trial, l)
                inst_layers.append(l.sample(**config))
            else:
                newl = copy.deepcopy(l)
                inst_layers.append(newl)
        return inst_layers

    def add(self, layer):
        self.layers_.append(layer)

    #@hpo.args()
    def compile(self, *args, **kwargs):
        self.compile_args = args
        self.compile_kwargs = kwargs

    def _model_init_args(self, trial):
        # for lazy model init
        # use backend to sample model init args
        inst_layers = self._istantiate_layers(trial, self.layers_)
        return {'layers':inst_layers, 'name': self.name_}

    def _model_compile(self, model, trial):
        # for lazy model compile
        # use backedn to sample compile args
        # config = OptunaBackend.sample_config(tiral, kwspaces)
        model.compile(*self.compile_args, **self.compile_kwargs)

    def fit(self, *args, **kwargs):
    #     if not self.tune_end:
    #         self.end_tune()
        super().fit(*args, **kwargs)

