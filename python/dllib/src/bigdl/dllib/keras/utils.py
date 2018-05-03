#
# Copyright 2018 Analytics Zoo Authors.
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

from bigdl.optim.optimizer import *
from bigdl.nn.criterion import *


def to_bigdl_optim_method(optimizer):
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
        raise TypeError("Unsupported optimizer: %s" % optimizer)


def to_bigdl_criterion(criterion):
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
        raise TypeError("Unsupported loss: %s" % criterion)


def to_bigdl_metrics(metrics):
    metrics = to_list(metrics)
    bmetrics = []
    for metric in metrics:
        if metric.lower() == "accuracy":
            bmetrics.append(Top1Accuracy())
        else:
            raise TypeError("Unsupported metrics: %s" % metric)
    return bmetrics
