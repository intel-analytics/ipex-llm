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


from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import clone_model
import tensorflow as tf
import inspect


def is_creator(model):
    return inspect.ismethod(model) or inspect.isfunction(model)

class Objective(object):
    def __init__(self,
                 model=None,
                 target_metric=None,
                 **kwargs,
                 ):
        """Init the objective.

        Args:
            model (keras model or function): a model instance or creator function. Defaults to None.
            model_compiler (function, optional): the compiler function. Defaults to None.
            target_metric (str, optional): target metric to optimize. Defaults to None.

        Raises:
            ValueError: _description_
        """
        if not is_creator(model) and not isinstance(model, tf.keras.Model) :
            raise ValueError("You should either pass a Tensorflo Keras model, or \
                            a model_creator to the Tuning objective.")

        self.model_ = model
        self.target_metric = target_metric
        self.kwargs = kwargs



    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()
        # TODO may add data creator here, e.g. refresh data, reset generators, etc.
        # create model
        if is_creator(self.model_):
            model = self.model_(trial)
        else:
            # copy model so that the original model is not changed
            # Need tests to check this path
            model = clone_model(self.model_)

        # fit
        hist = model.fit(**self.kwargs)

        # evaluate
        (x_valid, y_valid) = self.kwargs.get('validation_data', (None, None))
        if x_valid is not None:
            scores = model.evaluate(x_valid, y_valid, verbose=0)
        else:
            x_train = self.kwargs.get('x')
            y_train = self.kwargs.get('y')
            scores = model.evaluate(x_train, y_train, verbose=0)
            # return max(hist.history[self.target_metric])
        if self.target_metric is not None:
            try:
                metric_index = model.metrics_names.index(self.target_metric)
            except ValueError:
                raise ValueError("Target_metric should be one of the metrics \
                                specified in the compile metrics")
            score = scores[metric_index]
        else:
            score = scores[1]  # the first metric specified in compile

        return score
