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

import keras.optimizers as koptimizers

import bigdl.dllib.nn.criterion as bcriterion
import bigdl.dllib.optim.optimizer as boptimizer
from bigdl.dllib.utils.common import to_list
from bigdl.dllib.keras.converter import *
from keras.objectives import *
import six
from bigdl.dllib.utils.log4Error import *


class OptimConverter:

    @staticmethod
    def to_bigdl_metrics(metrics):
        metrics = to_list(metrics)
        bmetrics = []
        for metric in metrics:
            if metric == "accuracy":
                bmetrics.append(boptimizer.Top1Accuracy())
            else:
                unsupport_exp(metric)
        # TODO: add more metrics
        return bmetrics

    @staticmethod
    def to_bigdl_criterion(kloss):
        if isinstance(kloss, six.string_types):
            kloss = kloss.lower()
        if kloss == "categorical_crossentropy" or kloss == categorical_crossentropy:
            return bcriterion.CategoricalCrossEntropy()
        elif kloss == "mse" or kloss == "mean_squared_error" or kloss == mse:
            return bcriterion.MSECriterion()
        elif kloss == "binary_crossentropy" or kloss == binary_crossentropy:
            return bcriterion.BCECriterion()
        elif kloss == "mae" or kloss == "mean_absolute_error" or kloss == mae:
            return bcriterion.AbsCriterion()
        elif kloss == "hinge" or kloss == hinge:
            return bcriterion.MarginCriterion()
        elif kloss == "mean_absolute_percentage_error" or \
                kloss == "mape" or kloss == mean_absolute_percentage_error:
            return bcriterion.MeanAbsolutePercentageCriterion()
        elif kloss == "mean_squared_logarithmic_error" or \
                kloss == "msle" or kloss == mean_squared_logarithmic_error:
            return bcriterion.MeanSquaredLogarithmicCriterion()
        elif kloss == "squared_hinge" or kloss == squared_hinge:
            return bcriterion.MarginCriterion(squared=True)
        elif kloss == "sparse_categorical_crossentropy" or \
                kloss == sparse_categorical_crossentropy:
            return bcriterion.ClassNLLCriterion(logProbAsInput=False)
        elif kloss == "kullback_leibler_divergence" or \
                kloss == "kld" or kloss == kullback_leibler_divergence:
            return bcriterion.KullbackLeiblerDivergenceCriterion()
        elif kloss == "poisson" or kloss == poisson:
            return bcriterion.PoissonCriterion()
        elif kloss == "cosine_proximity" or kloss == "cosine" or kloss == cosine_proximity:
            return bcriterion.CosineProximityCriterion()
        else:
            invalidInputError(False, "Not supported loss: %s" % kloss)

    @staticmethod
    def to_bigdl_optim_method(koptim_method):
        # koptim_method is always an object
        lr = float(K.eval(koptim_method.lr))
        decay = float(K.eval(koptim_method.decay))
        if isinstance(koptim_method, koptimizers.Adagrad):
            warnings.warn("For Adagrad, we don't support epsilon for now")
            return boptimizer.Adagrad(learningrate=lr,
                                      learningrate_decay=decay)
        elif isinstance(koptim_method, koptimizers.SGD):
            momentum = float(K.eval(koptim_method.momentum))
            return boptimizer.SGD(learningrate=lr,
                                  learningrate_decay=decay,
                                  momentum=momentum,
                                  nesterov=koptim_method.nesterov)
        elif isinstance(koptim_method, koptimizers.Adam):
            beta1 = float(K.eval(koptim_method.beta_1))
            beta2 = float(K.eval(koptim_method.beta_2))
            return boptimizer.Adam(learningrate=lr,
                                   learningrate_decay=decay,
                                   beta1=beta1,
                                   beta2=beta2,
                                   epsilon=koptim_method.epsilon)
        elif isinstance(koptim_method, koptimizers.RMSprop):
            rho = float(K.eval(koptim_method.rho))
            return boptimizer.RMSprop(learningrate=lr,
                                      learningrate_decay=decay,
                                      decayrate=rho,
                                      epsilon=koptim_method.epsilon)
        elif isinstance(koptim_method, koptimizers.Adadelta):
            warnings.warn("For Adadelta, we don't support learning rate and learning rate decay for"
                          " now")
            return boptimizer.Adadelta(decayrate=koptim_method.rho,
                                       epsilon=koptim_method.epsilon)
        elif isinstance(koptim_method, koptimizers.Adamax):
            beta1 = float(K.eval(koptim_method.beta_1))
            beta2 = float(K.eval(koptim_method.beta_2))
            warnings.warn("For Adamax, we don't support learning rate decay for now")
            return boptimizer.Adamax(learningrate=lr,
                                     beta1=beta1,
                                     beta2=beta2,
                                     epsilon=koptim_method.epsilon)
        else:
            unsupport_exp(koptim_method)
