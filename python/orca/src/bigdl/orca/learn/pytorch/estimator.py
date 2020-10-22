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
from zoo.pipeline.estimator.estimator import Estimator as SparkEstimator
from zoo.orca.learn.pytorch.training_operator import TrainingOperator
from zoo.orca.data import SparkXShards
from bigdl.optim.optimizer import MaxEpoch
from zoo.feature.common import FeatureSet

import torch
from torch.optim.optimizer import Optimizer as TorchOptimizer
from torch.utils.data import DataLoader


class Estimator(object):
    def fit(self, data, epochs, **kwargs):
        pass

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, **kwargs):
        pass

    def get_model(self):
        pass

    def save(self, checkpoint):
        pass

    def load(self, checkpoint):
        pass

    def shutdown(self, force=False):
        pass

    @staticmethod
    def from_torch(*,
                   model,
                   optimizer,
                   loss=None,
                   scheduler_creator=None,
                   training_operator_cls=TrainingOperator,
                   initialization_hook=None,
                   config=None,
                   scheduler_step_freq="batch",
                   use_tqdm=False,
                   workers_per_node=1,
                   model_dir=None,
                   backend="bigdl"):
        if backend in {"horovod", "pytorch"}:
            return PyTorchRayEstimatorWrapper(model_creator=model,
                                              optimizer_creator=optimizer,
                                              loss_creator=loss,
                                              scheduler_creator=scheduler_creator,
                                              training_operator_cls=training_operator_cls,
                                              initialization_hook=initialization_hook,
                                              config=config,
                                              scheduler_step_freq=scheduler_step_freq,
                                              use_tqdm=use_tqdm,
                                              workers_per_node=workers_per_node,
                                              backend=backend)
        elif backend == "bigdl":
            return PytorchSparkEstimatorWrapper(model=model,
                                                loss=loss,
                                                optimizer=optimizer,
                                                model_dir=model_dir,
                                                bigdl_type="float")
        else:
            raise ValueError("only horovod, pytorch and bigdl backend are supported for now,"
                             f" got backend: {backend}")


class PyTorchRayEstimatorWrapper(Estimator):
    def __init__(self,
                 *,
                 model_creator,
                 optimizer_creator,
                 loss_creator=None,
                 scheduler_creator=None,
                 training_operator_cls=TrainingOperator,
                 initialization_hook=None,
                 config=None,
                 scheduler_step_freq="batch",
                 use_tqdm=False,
                 backend="pytorch",
                 workers_per_node=1):
        from zoo.orca.learn.pytorch.pytorch_ray_estimator import PyTorchRayEstimator
        self.estimator = PyTorchRayEstimator(model_creator=model_creator,
                                             optimizer_creator=optimizer_creator,
                                             loss_creator=loss_creator,
                                             scheduler_creator=scheduler_creator,
                                             training_operator_cls=training_operator_cls,
                                             initialization_hook=initialization_hook,
                                             config=config,
                                             scheduler_step_freq=scheduler_step_freq,
                                             use_tqdm=use_tqdm,
                                             backend=backend,
                                             workers_per_node=workers_per_node)

    def fit(self, data, epochs=1, profile=False, reduce_results=True, info=None):
        """

        :param data: (callable) a funtion that takes a config dict as input and return a data
            loader containing the training data.
        :param epochs: (int) Number of epochs to train the model
        :param profile: (bool) Returns time stats for the training procedure.
        :param reduce_results: (bool) Whether to average all metrics across all workers into one
            dict. If a metric is a non-numerical value (or nested dictionaries), one value will be
            randomly selected among the workers. If False, returns a list of dicts.
        :param info: (dict) Optional dictionary passed to the training operator for ``train_epoch``
            and ``train_batch``.
        :return: (list) A list of stats whose length will be equal to ``epochs``.
                stats is a dictionary of metrics for training.
                    You can provide custom metrics by passing in a custom
                    ``training_operator_cls``. If ``reduce_results=False``,
                    this will return a list of metric dictionaries whose
                    length will be equal to ``num_workers``.
        """
        return self.estimator.train(data_creator=data, epochs=epochs,
                                    profile=profile, reduce_results=reduce_results, info=info)

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, num_steps=None, profile=False, info=None):
        """

        :param data: (callable) a funtion that takes a config dict as input and return
            a data loader containing the validation data.
        :param num_steps: (int) Number of batches to compute update steps on.
               This corresponds also to the number of times ``TrainingOperator.validate_batch``
               is called.
        :param profile: (bool) Returns time stats for the evaluation procedure.
        :param info: (dict) Optional dictionary passed to the training operator for `validate`
            and `validate_batch`.
        :return: A dictionary of metrics for validation.
            You can provide custom metrics by passing in a custom ``training_operator_cls``.
        """
        return self.estimator.validate(data_creator=data, num_steps=num_steps, profile=profile,
                                       info=info)

    def get_model(self):
        """Returns the learned model(s)."""
        return self.estimator.get_model()

    def save(self, checkpoint):
        """Saves the Estimator state to the provided checkpoint path.

        :param checkpoint: (str) Path to target checkpoint file.
        """
        return self.estimator.save(checkpoint=checkpoint)

    def load(self, checkpoint):
        """Loads the Estimator and all workers from the provided checkpoint.

        :param checkpoint: (str) Path to target checkpoint file.
        """
        return self.estimator.load(checkpoint=checkpoint)

    def shutdown(self, force=False):
        """Shuts down workers and releases resources."""
        return self.estimator.shutdown(force=force)


class PytorchSparkEstimatorWrapper(Estimator):
    def __init__(self, model, loss, optimizer, model_dir=None, bigdl_type="float"):
        from zoo.pipeline.api.torch import TorchModel, TorchLoss, TorchOptim
        self.loss = loss
        if self.loss is None:
            self.loss = TorchLoss()
        else:
            self.loss = TorchLoss.from_pytorch(loss)
        if optimizer is None:
            from bigdl.optim.optimizer import SGD
            optimizer = SGD()
        elif isinstance(optimizer, TorchOptimizer):
            optimizer = TorchOptim.from_pytorch(optimizer)
        self.model_dir = model_dir
        self.model = TorchModel.from_pytorch(model)
        self.estimator = SparkEstimator(self.model, optimizer, model_dir, bigdl_type=bigdl_type)

    def fit(self, data, epochs=1, batch_size=32, validation_data=None, validation_methods=None,
            checkpoint_trigger=None):
        from zoo.orca.data.utils import to_sample
        from zoo.orca.learn.metrics import Metrics
        from zoo.orca.learn.trigger import Trigger

        end_trigger = MaxEpoch(epochs)
        assert batch_size > 0, "batch_size should be greater than 0"
        validation_methods = Metrics.convert_metrics_list(validation_methods)
        checkpoint_trigger = Trigger.convert_trigger(checkpoint_trigger)

        if isinstance(data, SparkXShards):
            train_rdd = data.rdd.flatMap(to_sample)
            train_feature_set = FeatureSet.sample_rdd(train_rdd)
            if validation_data is None:
                val_feature_set = None
            else:
                assert isinstance(validation_data, SparkXShards), "validation_data should be a " \
                                                                  "SparkXShards"
                val_feature_set = FeatureSet.sample_rdd(validation_data.rdd.flatMap(to_sample))

            self.estimator.train(train_feature_set, self.loss, end_trigger, checkpoint_trigger,
                                 val_feature_set, validation_methods, batch_size)
        elif isinstance(data, DataLoader) or callable(data):
            train_feature_set = FeatureSet.pytorch_dataloader(data, "", "")
            if validation_data is None:
                val_feature_set = None
            else:
                assert isinstance(validation_data, DataLoader) or callable(data), \
                    "validation_data should be a pytorch DataLoader or a callable data_creator"
                val_feature_set = FeatureSet.pytorch_dataloader(validation_data)

            self.estimator.train_minibatch(train_feature_set, self.loss, end_trigger,
                                           checkpoint_trigger, val_feature_set, validation_methods)
        else:
            raise ValueError("Data and validation data should be SparkXShards, DataLoaders or "
                             "callable data_creators but get " + data.__class__.__name__)
        return self

    def predict(self, data, **kwargs):
        pass

    def evaluate(self, data, validation_methods=None, batch_size=32):
        from zoo.orca.data.utils import to_sample
        from zoo.orca.learn.metrics import Metrics

        assert data is not None, "validation data shouldn't be None"
        validation_methods = Metrics.convert_metrics_list(validation_methods)

        if isinstance(data, SparkXShards):
            val_feature_set = FeatureSet.sample_rdd(data.rdd.flatMap(to_sample))
            return self.estimator.evaluate(val_feature_set, validation_methods, batch_size)
        elif isinstance(data, DataLoader) or callable(data):
            val_feature_set = FeatureSet.pytorch_dataloader(data)
            return self.estimator.evaluate_minibatch(val_feature_set, validation_methods)
        else:
            raise ValueError("Data should be a SparkXShards, a DataLoader or a callable "
                             "data_creator, but get " + data.__class__.__name__)

    def get_model(self):
        return self.model.to_pytorch()

    def save(self, checkpoint):
        pass

    def load(self, checkpoint, loss=None):
        from zoo.orca.learn.utils import find_latest_checkpoint
        from bigdl.nn.layer import Model
        from bigdl.optim.optimizer import OptimMethod
        import os
        if loss is not None:
            from zoo.pipeline.api.torch import TorchLoss
            self.loss = TorchLoss.from_pytorch(loss)
        path, prefix, version = find_latest_checkpoint(checkpoint, model_type="pytorch")
        if path is None:
            raise ValueError("Cannot find PyTorch checkpoint, please check your checkpoint path.")
        try:
            self.model = Model.load(os.path.join(path, "model.{}".format(version)))
            optimizer = OptimMethod.load(os.path.join(path, "{}.{}".format(prefix, version)))
        except Exception:
            raise ValueError("Cannot load PyTorch checkpoint, please check your checkpoint path "
                             "and checkpoint type.")
        self.estimator = SparkEstimator(self.model, optimizer, self.model_dir)

    def shutdown(self, force=False):
        pass

    def clear_gradient_clipping(self):
        """
        Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
        In order to take effect, it needs to be called before fit.
        :return:
        """
        self.estimator.clear_gradient_clipping()

    def set_constant_gradient_clipping(self, min, max):
        """
        Set constant gradient clipping during the training process.
        In order to take effect, it needs to be called before fit.
        :param min: The minimum value to clip by.
        :param max: The maximum value to clip by.
        :return:
        """
        self.estimator.set_constant_gradient_clipping(min=min, max=max)

    def set_l2_norm_gradient_clipping(self, clip_norm):
        """
        Clip gradient to a maximum L2-Norm during the training process.
        In order to take effect, it needs to be called before fit.
        :param clip_norm: Gradient L2-Norm threshold.
        :return:
        """
        self.estimator.set_l2_norm_gradient_clipping(clip_norm=clip_norm)
