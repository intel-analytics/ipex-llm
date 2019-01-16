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
from zoo.pipeline.api.keras.objectives import *
from zoo.pipeline.api.keras import metrics


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
        return MeanSquaredError()
    elif criterion == "binary_crossentropy":
        return BinaryCrossEntropy()
    elif criterion == "mae" or criterion == "mean_absolute_error":
        return mae()
    elif criterion == "hinge":
        return Hinge()
    elif criterion == "mean_absolute_percentage_error" or criterion == "mape":
        return MeanAbsolutePercentageError()
    elif criterion == "mean_squared_logarithmic_error" or criterion == "msle":
        return MeanSquaredLogarithmicError()
    elif criterion == "squared_hinge":
        return SquaredHinge()
    elif criterion == "sparse_categorical_crossentropy":
        return SparseCategoricalCrossEntropy()
    elif criterion == "kullback_leibler_divergence" or criterion == "kld":
        return KullbackLeiblerDivergence()
    elif criterion == "poisson":
        return Poisson()
    elif criterion == "cosine_proximity" or criterion == "cosine":
        return CosineProximity()
    elif criterion == "rank_hinge":
        return RankHinge()
    else:
        raise TypeError("Unsupported loss: %s" % criterion)


def to_bigdl_metric(metric):
    metric = metric.lower()
    if metric == "accuracy" or metric == "acc":
        return metrics.Accuracy()
    elif metric == "top5accuracy" or metric == "top5acc":
        return metrics.Top5Accuracy()
    elif metric == "mae":
        from bigdl.optim.optimizer import MAE
        return MAE()
    elif metric == "auc":
        return metrics.AUC()
    elif metric == "loss":
        return Loss()
    elif metric == "treennaccuracy":
        return TreeNNAccuracy()
    else:
        raise TypeError("Unsupported metric: %s" % metric)


def to_bigdl_metrics(metrics):
    return [to_bigdl_metric(m) for m in metrics]
