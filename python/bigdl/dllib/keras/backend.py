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

from bigdl.keras.optimization import *
from bigdl.util.common import *


class KerasModelWrapper:

    def __init__(self, kmodel):
        redire_spark_logs()
        show_bigdl_info_logs()
        self.bmodel = DefinitionLoader.from_kmodel(kmodel)
        WeightLoader.load_weights_from_kmodel(self.bmodel, kmodel)  # share the same weight.
        self.criterion = OptimConverter.to_bigdl_criterion(kmodel.loss) if kmodel.loss else None
        self.optim_method =\
            OptimConverter.to_bigdl_optim_method(kmodel.optimizer) if kmodel.optimizer else None
        self.metrics = OptimConverter.to_bigdl_metrics(kmodel.metrics) if kmodel.metrics else None

    def evaluate(self, x, y, batch_size=32, sample_weight=None, is_distributed=False):
        """
        Evaluate a model by the given metrics.
        :param x: ndarray or list of ndarray for local mode.
                  RDD[Sample] for distributed mode
        :param y: ndarray or list of ndarray for local mode and would be None for cluster mode.
        :param batch_size
        :param is_distributed: run in local mode or distributed mode.
               NB: if is_distributed=true, x should be RDD[Sample] and y should be None
        :return:
        """
        if sample_weight:
            unsupport_exp("sample_weight")
        if is_distributed:
            if isinstance(x, np.ndarray):
                input = to_sample_rdd(x, y)
            elif isinstance(x, RDD):
                input = x
            if self.metrics:
                sc = get_spark_context()
                return [r.result for r in
                        self.bmodel.evaluate(input, batch_size, self.metrics)]
            else:
                raise Exception("No Metrics found.")
        else:
            raise Exception("We only support evaluation in distributed mode")

    def predict(self, x, batch_size=None, verbose=None, is_distributed=False):
        """Generates output predictions for the input samples,
        processing the samples in a batched way.

        # Arguments
            x: the input data, as a Numpy array or list of Numpy array for local mode.
               as RDD[Sample] for distributed mode
            is_distributed: used to control run in local or cluster. the default value is False
        # Returns
            A Numpy array or RDD[Sample] of predictions.
        """
        if batch_size or verbose:
            raise Exception("we don't support batch_size or verbose for now")
        if is_distributed:
            if isinstance(x, np.ndarray):
                input = to_sample_rdd(x, np.zeros([x.shape[0]]))
            #  np.asarray(self.bmodel.predict(x_rdd).collect())
            elif isinstance(x, RDD):
                input = x
            return self.bmodel.predict(input)
        else:
            if isinstance(x, np.ndarray):
                return self.bmodel.predict_local(x)
        raise Exception("not supported type: %s" % x)

    def fit(self, x, y=None, batch_size=32, nb_epoch=10, verbose=1, callbacks=None,
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None, initial_epoch=0, is_distributed=False):
        """Optimize the model by the given options

        :param x: ndarray or list of ndarray for local mode.
                  RDD[Sample] for distributed mode
        :param y: ndarray or list of ndarray for local mode and would be None for cluster mode.
            is_distributed: used to control run in local or cluster. the default value is False.
            NB: if is_distributed=true, x should be RDD[Sample] and y should be None
        :param is_distributed: Whether to train in local mode or distributed mode
        :return:
            A Numpy array or RDD[Sample] of predictions.
        """
        if callbacks:
            raise Exception("We don't support callbacks in fit for now")
        if class_weight:
            unsupport_exp("class_weight")
        if sample_weight:
            unsupport_exp("sample_weight")
        if initial_epoch != 0:
            unsupport_exp("initial_epoch")
        if shuffle != True:
            unsupport_exp("shuffle")
        if validation_split != 0.:
            unsupport_exp("validation_split")
        bopt = self.__create_optimizer(x=x,
                                       y=y,
                                       batch_size=batch_size,
                                       nb_epoch=nb_epoch,
                                       validation_data=validation_data,
                                       is_distributed=is_distributed)
        bopt.optimize()

    def __create_optimizer(self, x=None, y=None, batch_size=32, nb_epoch=10,
                           validation_data=None, is_distributed=False):
        if is_distributed:
            if isinstance(x, np.ndarray):
                input = to_sample_rdd(x, y)
                validation_data_rdd = to_sample_rdd(*validation_data)
            elif isinstance(x, RDD):
                input = x
                validation_data_rdd = validation_data
            return self.__create_distributed_optimizer(training_rdd=input,
                                                       batch_size=batch_size,
                                                       nb_epoch=nb_epoch,
                                                       validation_data=validation_data_rdd)
        else:
            if isinstance(x, np.ndarray):
                return self.__create_local_optimizer(x, y,
                                                     batch_size=batch_size,
                                                     nb_epoch=nb_epoch,
                                                     validation_data=validation_data)
        raise Exception("not supported type: %s" % x)

    def __create_local_optimizer(self, x, y, batch_size=32, nb_epoch=10, validation_data=None):
        if validation_data:
            raise unsupport_exp("validation_data")
        bopt = boptimizer.LocalOptimizer(
            X=x,
            Y=y,
            model=self.bmodel,
            criterion=self.criterion,
            end_trigger=boptimizer.MaxEpoch(nb_epoch),
            batch_size=batch_size,
            optim_method=self.optim_method,
            cores=None
        )
        # TODO: enable validation for local optimizer.
        return bopt

    def __create_distributed_optimizer(self, training_rdd,
                                       batch_size=32,
                                       nb_epoch=10,
                                       validation_data=None):
        sc = get_spark_context()
        bopt = boptimizer.Optimizer(
            model=self.bmodel,
            training_rdd=training_rdd,
            criterion=self.criterion,
            end_trigger=boptimizer.MaxEpoch(nb_epoch),
            batch_size=batch_size,
            optim_method=self.optim_method
        )
        if validation_data:
            bopt.set_validation(batch_size,
                                val_rdd=validation_data,
                                # TODO: check if keras use the same strategy
                                trigger=boptimizer.EveryEpoch(),
                                val_method=self.metrics)
        return bopt


def with_bigdl_backend(kmodel):
    init_engine()
    return KerasModelWrapper(kmodel)

