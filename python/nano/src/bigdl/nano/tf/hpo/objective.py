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


class Objective(object):
    def __init__(self,
                 model_instance=None,
                 model_cls=None,
                 model_initor=None,
                 model_compiler=None,
                 target_metric=None,
                 *args,
                 **kwargs,
                 ):
        """Init the objective

        Args:
            model_instance (keras model, optional):  the created model instance. Defaults to None.
            model_initor (closure, optional):  function to create the input args. Defaults to None
            model_cls (class type, optional): The model class used with model_initor to create model.
                            e.g. tf.Keras.Sequential, tf.Keras.Model
               Either model_instance or model_initor should be a non-None value.
               if both not None, use model_initor and ignore the other.
            model_compiler (closure, optional): model compile function. Defaults to None.

        Raises:
            ValueError: raised when both model_instance and model_initor is None.
        """
        if model_instance is None and model_initor is None:
            raise ValueError("You should either pass a created model, or \
                             a model_init to the Tuning objective.")

        self.model = model_instance
        self.model_initor = model_initor
        if self.model is not None and self.model_initor is not None:
            if model_cls is None:
                raise ValueError(
                    "model_cls should also be specified if smodel_initor is specified")
            self.model = None
            print("Warn: passed model is ignored when model_init is not null")

        self.model_cls = model_cls

        self.model_compiler = model_compiler
        self.target_metric = target_metric
        self.args = args
        self.kwargs = kwargs

    def __call__(self, trial):
        # Clear clutter from previous Keras session graphs.
        clear_session()
        # TODO may add data creator here, e.g. refresh data, reset generators, etc.
        # create model
        if self.model is not None:
            # copy model so that the original model is not changed
            # Need tests to check this path
            model = clone_model(self.model)
        else:
            assert(self.model_cls is not None)
            model = self.model_cls(**self.model_initor(trial))

        # compile
        self.model_compiler(model, trial)
        # fit
        hist = model.fit(*self.args, **self.kwargs)

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
