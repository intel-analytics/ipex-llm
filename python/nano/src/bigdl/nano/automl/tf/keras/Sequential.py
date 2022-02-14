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
from bigdl.nano.automl.utils.lazyutils import proxy_methods
import bigdl.nano.tf.keras.Sequential as NanoSequential
from bigdl.nano.automl.hpo.mixin import HPOMixin
from bigdl.nano.automl.hpo.backend import OptunaBackend
import bigdl.nano.automl.hpo as hpo
import copy


@proxy_methods
class Sequential(HPOMixin, tf.keras.Sequential):

    # these methods are automatically created using "@proxy_methods"
    # details see desriptions in _proxy method
    PROXYED_METHODS = ['predict', 'predict_on_batch',
            'evaluate', 'test_on_batch',
            'to_json', 'to_yaml', 'summary',
            'save', 'save_spec', 'save_weights',
            'get_layer']


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
        super().__init__(layers, name)
        self._lazymodel = None


    def _proxy(self, name, method, *args, **kwargs):
        # call to keras method is forwarded to internal model
        # NOTE: keep the unused "method" argument so that
        # only the methods which are actually called are created
        if not self._lazymodel:
            raise ValueError("Model is not actually built yet. "+ \
                "Please call end_search before calling \""+name+"\"")
        internal_m = getattr(self._lazymodel, name)
        return internal_m(*args, **kwargs)

    def _instantiate_layers(self,trial, layers):
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
        inst_layers = self._instantiate_layers(trial, self.layers_)
        return {'layers':inst_layers, 'name': self.name_}

    def _model_compile(self, model, trial):
        # for lazy model compile
        # use backedn to sample compile args
        # config = OptunaBackend.sample_config(trial, kwspaces)
        model.compile(*self.compile_args, **self.compile_kwargs)

    def _model_build(self, trial):
        # for lazy model build
        # build model based on searched hyperparams from trial
        # TODO may add data creator here, e.g. refresh data, reset generators, etc.
        #super().__init__(**self._model_init_args(trial))
        #self._model_compile(super(), trial)
        # use composition instead of inherited
        modelcls = self.__class__.__bases__[1]
        self._lazymodel = modelcls(**self._model_init_args(trial))
        self._model_compile(self._lazymodel, trial)

    def fit(self, *args, **kwargs):
        if not self.tune_end:
             self.end_search()
        self._lazymodel.fit(*args, **kwargs)




