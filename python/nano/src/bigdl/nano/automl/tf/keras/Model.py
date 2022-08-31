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

from bigdl.nano.automl.utils import proxy_methods
from bigdl.nano.automl.tf.mixin import HPOMixin
from bigdl.nano.automl.hpo.callgraph import CallCache


@proxy_methods
class Model(HPOMixin, tf.keras.Model):
    """Tf.keras.Model with HPO capabilities."""

    def __init__(self, **kwargs):
        """Initializer."""
        # we only take keyword arguments for now
        # TODO check how args is used
        # TODO check why base class is keras.engine.training_v1.Model
        super().__init__()
        self.model_class = tf.keras.Model
        self.kwargs = kwargs
        self.lazyinputs_ = kwargs.get('inputs', None)
        self.lazyoutputs_ = kwargs.get('outputs', None)

    def _model_init_args(self, trial):
        # for lazy model init
        # use backend to sample model init args
        # and construct the actual layers
        in_tensors, out_tensors = CallCache.execute(
            self.lazyinputs_,
            self.lazyoutputs_,
            trial,
            self.backend)
        self.kwargs['inputs'] = in_tensors
        self.kwargs['outputs'] = out_tensors
        return self.kwargs

    def _get_model_init_args_func_kwargs(self):
        """Return the kwargs of _model_init_args_func except trial."""
        return {
            'lazyinputs': self.lazyinputs_,
            'lazyoutputs': self.lazyoutputs_,
            'kwargs': self.kwargs,
            'backend': self.backend
        }

    @staticmethod
    def _model_init_args_func(trial, lazyinputs, lazyoutputs, kwargs, backend):
        in_tensors, out_tensors = CallCache.execute(
            lazyinputs,
            lazyoutputs,
            trial,
            backend)
        kwargs['inputs'] = in_tensors
        kwargs['outputs'] = out_tensors
        return kwargs
