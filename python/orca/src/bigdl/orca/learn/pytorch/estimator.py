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

import logging
from bigdl.orca.learn.pytorch.training_operator import TrainingOperator


class Estimator(object):
    @staticmethod
    def from_torch(*,
                   model,
                   optimizer,
                   loss=None,
                   metrics=None,
                   scheduler_creator=None,
                   training_operator_cls=TrainingOperator,
                   initialization_hook=None,
                   config=None,
                   scheduler_step_freq="batch",
                   use_tqdm=False,
                   workers_per_node=1,
                   model_dir=None,
                   backend="bigdl",
                   sync_stats=False,
                   log_level=logging.INFO,
                   log_to_driver=True,
                   ):
        """
        Create an Estimator for torch.

        :param model: PyTorch model or model creator function if backend="bigdl", PyTorch
               model creator function if backend="horovod" or "ray"
        :param optimizer: Orca/PyTorch optimizer or optimizer creator function if backend="bigdl"
               , PyTorch optimizer creator function if backend="horovod" or "ray"
        :param loss: PyTorch loss or loss creator function if backend="bigdl", PyTorch loss creator
               function if backend="horovod" or "ray"
        :param metrics: Orca validation methods for evaluate.
        :param scheduler_creator: parameter for `horovod` and `ray` backends. a
               learning rate scheduler wrapping the optimizer. You will need to set
               ``scheduler_step_freq="epoch"`` for the scheduler to be incremented correctly.
        :param config: parameter config dict, CfgNode or any class that plays a role of
               configuration to create model, optimizer loss and data.
        :param scheduler_step_freq: parameter for `horovod` and `ray` backends.
               "batch", "epoch" or None. This will determine when ``scheduler.step`` is called. If
               "batch", ``step`` will be called after every optimizer step. If "epoch", ``step``
               will be called after one pass of the DataLoader. If a scheduler is passed in, this
               value is expected to not be None.
        :param use_tqdm: parameter for `horovod` and `ray` backends. You can monitor
               training progress if use_tqdm=True.
        :param workers_per_node: parameter for `horovod` and `ray` backends. worker
               number on each node. default: 1.
        :param model_dir: parameter for `bigdl` and `spark` backend. The path to save model. During
               the training, if checkpoint_trigger is defined and triggered, the model will be
               saved to model_dir.
        :param backend: You can choose "horovod",  "ray", "bigdl" or "spark" as
               backend. Default: `bigdl`.
        :param sync_stats: Whether to sync metrics across all distributed workers after each epoch.
               If set to False, only rank 0's metrics are printed. This param only works horovod,
               ray and pyspark backend. For spark backend, the metrics printed are
               are always synced. This param only affects the printed metrics, the returned metrics
               are always averaged across workers. Default: True
        :param log_level: Setting the log_level of each distributed worker. This param only works
               horovod, ray and pyspark backend.
        :param log_to_driver: (bool) Whether display executor log on driver in cluster mode.
               Default: True. This option is only for "spark" backend.
        :return: an Estimator object.
        """
        if backend in {"horovod", "ray"}:
            from bigdl.orca.learn.pytorch.pytorch_ray_estimator import PyTorchRayEstimator
            return PyTorchRayEstimator(model_creator=model,
                                       optimizer_creator=optimizer,
                                       loss_creator=loss,
                                       metrics=metrics,
                                       scheduler_creator=scheduler_creator,
                                       training_operator_cls=training_operator_cls,
                                       initialization_hook=initialization_hook,
                                       config=config,
                                       scheduler_step_freq=scheduler_step_freq,
                                       use_tqdm=use_tqdm,
                                       workers_per_node=workers_per_node,
                                       backend=backend,
                                       sync_stats=sync_stats,
                                       log_level=log_level)
        elif backend == "bigdl":
            from bigdl.orca.learn.pytorch.pytorch_spark_estimator import PyTorchSparkEstimator
            return PyTorchSparkEstimator(model=model,
                                         loss=loss,
                                         optimizer=optimizer,
                                         config=config,
                                         metrics=metrics,
                                         model_dir=model_dir,
                                         bigdl_type="float")
        elif backend == "spark":
            from bigdl.orca.learn.pytorch.pytorch_pyspark_estimator import PyTorchPySparkEstimator
            return PyTorchPySparkEstimator(model_creator=model,
                                           optimizer_creator=optimizer,
                                           loss_creator=loss,
                                           metrics=metrics,
                                           scheduler_creator=scheduler_creator,
                                           training_operator_cls=training_operator_cls,
                                           initialization_hook=initialization_hook,
                                           config=config,
                                           scheduler_step_freq=scheduler_step_freq,
                                           use_tqdm=use_tqdm,
                                           workers_per_node=workers_per_node,
                                           sync_stats=sync_stats,
                                           log_level=log_level,
                                           model_dir=model_dir,
                                           log_to_driver=log_to_driver,
                                           )
        else:
            from bigdl.dllib.utils.log4Error import invalidInputError
            invalidInputError(False,
                              "Only horovod, ray, bigdl and spark backends are "
                              f"supported for now, got backend: {backend}")

    @staticmethod
    def latest_checkpoint(checkpoint_dir):
        from .callbacks.model_checkpoint import ModelCheckpoint
        checkpoint_path = ModelCheckpoint.get_latest_checkpoint(checkpoint_dir)
        return checkpoint_path
