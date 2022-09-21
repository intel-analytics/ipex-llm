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
from bigdl.orca.automl.model.abstract import BaseModel, ModelBuilder
import numpy as np
from bigdl.orca.automl.metrics import Evaluator
import pickle
import copy
import tensorflow as tf
from tensorflow.keras import backend as K
import types
from bigdl.dllib.utils.log4Error import *


def check_tf_version():
    tf_version = tf.__version__
    if tf_version >= "2.0":
        invalidInputError(False,
                          f"Currently running TensorFlow version {tf_version}. We only support"
                          f"TensorFlow 1.x for now and has been tested on 1.15")


class KerasBaseModel(BaseModel):
    def __init__(self,
                 model_creator,
                 check_optional_config=False):
        self.check_optional_config = check_optional_config
        self.model_creator = model_creator
        self.model = None
        self.config = None
        self.model_built = False

    def build(self, config):
        # update config
        self._check_config(**config)
        self.config = config
        # build model
        # TODO: move this to Chronos
        if "selected_features" in config:
            config["input_feature_num"] = len(config['selected_features'])\
                + config['output_feature_num']
        self.model = self.model_creator(config)
        # check whether the model is a compiled model
        if self.model.optimizer is None:
            invalidInputError(False,
                              "You must create a compiled model in model_creator")
        self.model_built = True

    @staticmethod
    def _np_to_dataset(data, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.batch(batch_size)
        return dataset

    def fit_eval(self, data, validation_data=None, mc=False, verbose=0, epochs=1, metric=None,
                 metric_func=None, resources_per_trial=None,
                 **config):
        """
        :param data: could be a tuple with numpy ndarray with form (x, y) or
               a data creator takes a config dict as parameter and returns a tf.data.Dataset.
        :param validation_data: could be a tuple with numpy ndarray with form (x, y)
        fit_eval will build a model at the first time it is built
        config will be updated for the second or later times with only non-model-arch
        params be functional
        TODO: check the updated params and decide if the model is needed to be rebuilt
        """
        def update_config():
            if isinstance(data, tuple) and isinstance(data[0], np.ndarray):
                x, y = data
                config.setdefault("input_dim", x.shape[-1])
                config.setdefault("output_dim", y.shape[-1])
                if metric and not metric_func:
                    config.update({"metric": metric})

        if not self.model_built:
            update_config()
            self.build(config)
        else:
            tmp_config = copy.copy(self.config)
            tmp_config.update(config)
            self._check_config(**tmp_config)
            self.config.update(config)

        # get train_dataset and validation_dataset
        if isinstance(data, types.FunctionType):
            train_dataset = data(self.config)
            if validation_data:
                validation_dataset = validation_data(self.config)
            else:
                validation_dataset = validation_data
        else:
            if not isinstance(data, tuple):
                invalidInputError(False,
                                  f"data/validation_data should be a tuple of numpy array "
                                  f"or a data creator function but found {type(data)}")
            if validation_data:
                invalidInputError(isinstance(validation_data, tuple),
                                  f"validation_data should be a tuple or"
                                  f" data creator function but found {type(validation_data)}")

            batch_size = int(self.config.get("batch_size", 32))
            train_dataset = KerasBaseModel._np_to_dataset(data, batch_size=batch_size)
            if validation_data:
                validation_dataset = KerasBaseModel._np_to_dataset(validation_data, batch_size)
            else:
                validation_dataset = validation_data

        hist = self.model.fit(train_dataset,
                              validation_data=validation_dataset,
                              epochs=epochs,
                              verbose=verbose
                              )

        # model.metrics_names are available only after a keras model has been trained/evaluated
        compiled_metric_names = self.model.metrics_names.copy()
        compiled_metric_names.remove("loss")
        if not metric_func:
            # check input metric value
            if not metric:
                if len(compiled_metric_names) == 1:
                    metric = compiled_metric_names[0]
                    metric_name = metric
                else:
                    invalidInputError(False,
                                      f"Got multiple metrics in compile: {compiled_metric_names}. "
                                      f"Please choose one target metric for automl optimization")
            else:
                if metric in compiled_metric_names:
                    metric_name = metric
                else:
                    try:
                        hist_metric_name = tf.keras.metrics.get(metric).__name__
                    except:
                        invalidInputError(False,
                                          f"get invalid metric name {metric} for tf.keras")
                    if hist_metric_name in compiled_metric_names:
                        metric_name = hist_metric_name
                    else:
                        invalidInputError(False,
                                          f"Input metric in fit_eval should be one of the metrics "
                                          f"that are used to compile the model. Got metric value"
                                          f" of {metric} and the metrics in compile are "
                                          f"{compiled_metric_names}")
            if validation_data is None:
                result = hist.history.get(metric_name)[-1]
            else:
                result = hist.history.get('val_' + metric_name)[-1]
            return {metric: result}
        else:
            metric_name = metric or metric_func.__name__
            if validation_data is not None:
                val_x = validation_data[0]
                val_y = validation_data[1]
            else:
                val_x = data[0]
                val_y = data[1]
            y_pred = self.predict(val_x)
            result = metric_func(val_y, y_pred)
            return {metric_name: result}

    def evaluate(self, x, y, batch_size=32, metrics=['mse'], multioutput='raw_values'):
        """
        Evaluate on x, y
        :param x: input
        :param y: target
        :param metrics: a list of metrics in string format
        :param multioutput: output mode
        :return: a list of metric evaluation results
        """
        y_pred = self.predict(x, batch_size=batch_size)
        return [Evaluator.evaluate(m, y, y_pred, multioutput=multioutput) for m in metrics]

    def predict(self, x, batch_size=32):
        """
        Prediction on x.
        :param x: input
        :param batch_size: batch
        :return: predicted y
        """
        if not self.model_built:
            invalidInputError(False,
                              "You must call fit_eval or restore first before calling predict!")
        return self.model.predict(x, batch_size=batch_size)

    def predict_with_uncertainty(self, x, n_iter=100):
        if not self.model_built:
            invalidInputError(False,
                              "You must call fit_eval or restore first before calling predict!")
        check_tf_version()
        f = K.function([self.model.layers[0].input, K.learning_phase()],
                       [self.model.layers[-1].output])
        result = np.array([f((x, 1))[0] for _ in range(n_iter)])

        prediction = result.mean(axis=0)
        uncertainty = result.var(axis=0)
        return prediction, uncertainty

    def state_dict(self):
        state = {
            "config": self.config,
            "weights": self.model.get_weights(),
            # "optimizer_weights": self.model.optimizer.get_weights()
        }
        return state

    def load_state_dict(self, state):
        self.config = state["config"]
        self.model = self.model_creator(self.config)
        self.model.set_weights(state["weights"])
        self.model_built = True
        # self.model.optimizer.set_weights(state["optimizer_weights"])

    def save(self, checkpoint):
        if not self.model_built:
            invalidInputError(False,
                              "You must call fit_eval or restore first before calling save!")
        state_dict = self.state_dict()
        with open(checkpoint, "wb") as f:
            pickle.dump(state_dict, f)

    def restore(self, checkpoint):
        with open(checkpoint, "rb") as f:
            state_dict = pickle.load(f)
        self.load_state_dict(state_dict)

    def _get_required_parameters(self):
        return set()

    def _get_optional_parameters(self):
        return {"batch_size"}


class KerasModelBuilder(ModelBuilder):

    def __init__(self, model_creator):
        self.model_creator = model_creator

    def build(self, config):
        model = KerasBaseModel(self.model_creator)
        model.build(config)
        return model
