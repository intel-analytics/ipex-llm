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


from selectors import EpollSelector
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import clone_model
import tensorflow as tf
import inspect
import copy

from bigdl.nano.automl.hpo.backend import create_tfkeras_pruning_callback
from bigdl.nano.utils.log4Error import invalidInputError


def _is_creator(model):
    return inspect.ismethod(model) or inspect.isfunction(model)


class Objective(object):
    """The Tuning objective for HPO."""

    def __init__(self,
                 model=None,
                 target_metric=None,
                 pruning=False,
                 backend=None,
                 report_method=None,
                 **kwargs
                 ):
        """
        Init the objective.

        :param: model: a model instance or a creator function.
            Defaults to None.
        :param: target_metric: str(optional): target metric to optimize.
            Defaults to None.
        :param: pruning: bool (optional): whether to enable pruning.
            Defaults to False.
        :param: backend: the HPO backend
        :param: report_method: a function to decide which score to report
            from the result of all epochs if there's more than one epoch
        throw: ValueError: _description_
        """
        if not _is_creator(model) and not isinstance(model, tf.keras.Model):
            invalidInputError(False,
                              "You should either pass a Tensorflo Keras model, or "
                              "a model_creator to the Tuning objective.")

        self.model_ = model
        self.target_metric_ = target_metric
        self.pruning = pruning
        self.backend = backend
        self.kwargs = kwargs

        if report_method is None:
            report_method = max
        self.report_method = report_method

    @property
    def target_metric(self):
        """Get the target metric."""
        return self.target_metric_

    @target_metric.setter
    def target_metric(self, value):
        """Set the target metric."""
        # TODO add more validity check here
        self.target_metric_ = value

    def _prepare_fit_args(self, trial):
        # only do shallow copy and process/duplicate
        # specific args TODO: may need to handle more cases
        new_kwargs = copy.copy(self.kwargs)
        new_kwargs['verbose'] = 2

        # process batch size
        new_kwargs = self.backend.instantiate_param(trial, new_kwargs, 'batch_size')

        # process callbacks
        callbacks = new_kwargs.get('callbacks', None)
        callbacks = callbacks() if inspect.isfunction(callbacks) else callbacks

        if self.pruning:
            callbacks = callbacks or []
            prune_callback = create_tfkeras_pruning_callback(trial, self.target_metric)
            callbacks.append(prune_callback)

        new_kwargs['callbacks'] = callbacks
        return new_kwargs

    def __call__(self, trial):
        """
        Execute Training and return target metric in each trial.

        :param: trial: the trial object which provides the hyperparameter combinition.
        :return: the target metric value.
        """
        # Clear clutter from previous Keras session graphs.
        clear_session()
        # TODO may add data creator here, e.g. refresh data, reset generators, etc.
        # create model
        if _is_creator(self.model_):
            model = self.model_(trial)
        else:
            # copy model so that the original model is not changed
            # Need tests to check this path
            model = clone_model(self.model_)

        # fit
        new_kwargs = self._prepare_fit_args(trial)
        hist = model.fit(**new_kwargs)

        score = hist.history.get(self.target_metric, None)
        if score is not None:
            if isinstance(score, list):
                # score = score[-1]
                score = self.report_method(score)
            return score
