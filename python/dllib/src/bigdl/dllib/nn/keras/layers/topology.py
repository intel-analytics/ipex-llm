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

from bigdl.dllib.nn.keras.layers.layer import KerasLayer
from bigdl.dllib.optim.optimizer import *
from bigdl.dllib.nn.criterion import *
from bigdl.dllib.utils.log4Error import *
import multiprocessing
import warnings

from bigdl.dllib.nn.layer import SharedStaticUtils, Container


class KerasModel(KerasLayer, Container, SharedStaticUtils):
    """
    .. note:: `bigdl.dllib.keras` is deprecated in 0.11.
    This will be removed in future releases.
    """

    def __convert_optim_method(self, optimizer):
        optimizer = optimizer.lower()
        if optimizer == "adagrad":
            return Adagrad(learningrate=0.01)
        elif optimizer == "sgd":
            return SGD(learningrate=0.01)
        elif optimizer == "adam":
            return Adam()
        elif optimizer == "rmsprop":
            return RMSprop(learningrate=0.001, decayrate=0.9)
        elif optimizer == "adadelta":
            return Adadelta(decayrate=0.95, epsilon=1e-8)
        elif optimizer == "adamax":
            return Adamax(epsilon=1e-8)
        else:
            invalidInputError(False, "Unsupported optimizer: %s" % optimizer)

    def __convert_criterion(self, criterion):
        criterion = criterion.lower()
        if criterion == "categorical_crossentropy":
            return CategoricalCrossEntropy()
        elif criterion == "mse" or criterion == "mean_squared_error":
            return MSECriterion()
        elif criterion == "binary_crossentropy":
            return BCECriterion()
        elif criterion == "mae" or criterion == "mean_absolute_error":
            return AbsCriterion()
        elif criterion == "hinge":
            return MarginCriterion()
        elif criterion == "mean_absolute_percentage_error" or criterion == "mape":
            return MeanAbsolutePercentageCriterion()
        elif criterion == "mean_squared_logarithmic_error" or criterion == "msle":
            return MeanSquaredLogarithmicCriterion()
        elif criterion == "squared_hinge":
            return MarginCriterion(squared=True)
        elif criterion == "sparse_categorical_crossentropy":
            return ClassNLLCriterion(logProbAsInput=False)
        elif criterion == "kullback_leibler_divergence" or criterion == "kld":
            return KullbackLeiblerDivergenceCriterion()
        elif criterion == "poisson":
            return PoissonCriterion()
        elif criterion == "cosine_proximity" or criterion == "cosine":
            return CosineProximityCriterion()
        else:
            invalidInputError(False, "Unsupported loss: %s" % criterion)

    def __convert_metrics(self, metrics):
        metrics = to_list(metrics)
        bmetrics = []
        for metric in metrics:
            if metric.lower() == "accuracy":
                bmetrics.append(Top1Accuracy())
            else:
                invalidInputError(False, "Unsupported metrics: %s" % metric)
        return bmetrics

    def compile(self, optimizer, loss, metrics=None):
        """
        Configures the learning process. Must be called before fit or evaluate.

        # Arguments
        optimizer: Optimization method to be used. One can alternatively pass in the corresponding
                   string representation, such as 'sgd'.
        loss: Criterion to be used. One can alternatively pass in the corresponding string
              representation, such as 'mse'.
        metrics: List of validation methods to be used. Default is None. One can alternatively use
        ['accuracy'].
        """
        if isinstance(optimizer, six.string_types):
            optimizer = self.__convert_optim_method(optimizer)
        if isinstance(loss, six.string_types):
            loss = self.__convert_criterion(loss)
        if all(isinstance(metric, six.string_types) for metric in metrics):
            metrics = self.__convert_metrics(metrics)
        callBigDlFunc(self.bigdl_type, "compile",
                      self.value,
                      optimizer,
                      loss,
                      metrics)

    def fit(self, x, y=None, batch_size=32, nb_epoch=10, validation_data=None, distributed=True):
        """
        Train a model for a fixed number of epochs on a dataset.

        # Arguments
        x: Input data. A Numpy array or RDD of Sample or Image DataSet.
        y: Labels. A Numpy array. Default is None if x is already RDD of Sample or Image DataSet.
        batch_size: Number of samples per gradient update.
        nb_epoch: Number of iterations to train.
        validation_data: Tuple (x_val, y_val) where x_val and y_val are both Numpy arrays.
                         Or RDD of Sample. Default is None if no validation is involved.
        distributed: Boolean. Whether to train the model in distributed mode or local mode.
                     Default is True. In local mode, x and y must both be Numpy arrays.
        """
        if distributed:
            if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
                training_data = to_sample_rdd(x, y)
                if validation_data:
                    validation_data = to_sample_rdd(*validation_data)
            elif (isinstance(x, RDD) or isinstance(x, DataSet)) and not y:
                training_data = x
            else:
                invalidInputError(False, "Unsupported training data type: %s" % type(x))
            callBigDlFunc(self.bigdl_type, "fit",
                          self.value,
                          training_data,
                          batch_size,
                          nb_epoch,
                          validation_data)
        else:
            if validation_data:
                val_x = [JTensor.from_ndarray(x) for x in to_list(validation_data[0])]
                val_y = JTensor.from_ndarray(validation_data[1])
            else:
                val_x, val_y = None, None
            callBigDlFunc(self.bigdl_type, "fit",
                          self.value,
                          [JTensor.from_ndarray(x) for x in to_list(x)],
                          JTensor.from_ndarray(y),
                          batch_size,
                          nb_epoch,
                          val_x,
                          val_y,
                          multiprocessing.cpu_count())

    def evaluate(self, x, y=None, batch_size=32):
        """
        Evaluate a model on a given dataset in distributed mode.

        # Arguments
        x: Input data. A Numpy array or RDD of Sample.
        y: Labels. A Numpy array. Default is None if x is already RDD of Sample.
        batch_size: Number of samples per gradient update.
        """
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            evaluation_data = to_sample_rdd(x, y)
        elif isinstance(x, RDD) and not y:
            evaluation_data = x
        else:
            invalidInputError(False, "Unsupported evaluation data type: %s" % type(x))
        return callBigDlFunc(self.bigdl_type, "evaluate",
                             self.value,
                             evaluation_data,
                             batch_size)

    def predict(self, x, distributed=True):
        """
        Use a model to do prediction.

        # Arguments
        x: Input data. A Numpy array or RDD of Sample.
        distributed: Boolean. Whether to do prediction in distributed mode or local mode.
                     Default is True. In local mode, x must be a Numpy array.
        """
        if is_distributed:
            if isinstance(x, np.ndarray):
                features = to_sample_rdd(x, np.zeros([x.shape[0]]))
            elif isinstance(x, RDD):
                features = x
            else:
                invalidInputError(False, "Unsupported prediction data type: %s" % type(x))
            return self.predict_distributed(features)
        else:
            if isinstance(x, np.ndarray):
                return self.predict_local(x)
            else:
                invalidInputError(False, "Unsupported prediction data type: %s" % type(x))


class Sequential(KerasModel):
    """
    Container for a sequential model.

    # Arguments
    name: String to specify the name of the sequential model. Default is None.

    >>> sequential = Sequential(name="seq1")
    creating: createKerasSequential

    .. note:: `bigdl.dllib.keras` is deprecated in 0.11.
    This will be removed in future releases.
    """

    def __init__(self, jvalue=None, **kwargs):
        warnings.warn("bigdl.dllib.nn.keras is deprecated. "
                      "Recommend to use bigdl.dllib.keras instead.")
        super(Sequential, self).__init__(jvalue, **kwargs)

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Sequential(jvalue=jvalue)
        model.value = jvalue
        return model

    def add(self, model):
        self.value.add(model.value)
        return self


class Model(KerasModel):
    """
    Container for a graph model.

    # Arguments
    input: An input node or a list of input nodes.
    output: An output node or a list of output nodes.
    name: String to specify the name of the graph model. Default is None.

    .. note:: `bigdl.dllib.keras` is deprecated in 0.11.
    This will be removed in future releases.
    """

    def __init__(self, input, output, jvalue=None, **kwargs):
        warnings.warn("bigdl.dllib.nn.keras is deprecated. "
                      "Recommend to use bigdl.dllib.keras instead.")
        super(Model, self).__init__(jvalue,
                                    to_list(input),
                                    to_list(output),
                                    **kwargs)

    @staticmethod
    def from_jvalue(jvalue, bigdl_type="float"):
        """
        Create a Python Model base on the given java value
        :param jvalue: Java object create by Py4j
        :return: A Python Model
        """
        model = Model([], [], jvalue=jvalue)
        model.value = jvalue
        return model
