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
                   cluster=None,
                   backend="bigdl"):
        """
        Create an Estimator for torch.

        :param model: PyTorch model or model creator function if backend="bigdl", PyTorch
               model creator function if backend="horovod" or "torch_distributed"
        :param optimizer: Orca/PyTorch optimizer or optimizer creator function if backend="bigdl"
               , PyTorch optimizer creator function if backend="horovod" or "torch_distributed"
        :param loss: PyTorch loss or loss creator function if backend="bigdl", PyTorch loss creator
               function if backend="horovod" or "torch_distributed"
        :param metrics: Orca validation methods for evaluate.
        :param scheduler_creator: parameter for `horovod` and `torch_distributed` backends. a
               learning rate scheduler wrapping the optimizer. You will need to set
               ``scheduler_step_freq="epoch"`` for the scheduler to be incremented correctly.
        :param config: parameter config dict to create model, optimizer loss and data.
        :param scheduler_step_freq: parameter for `horovod` and `torch_distributed` backends.
               "batch", "epoch" or None. This will determine when ``scheduler.step`` is called. If
               "batch", ``step`` will be called after every optimizer step. If "epoch", ``step``
               will be called after one pass of the DataLoader. If a scheduler is passed in, this
               value is expected to not be None.
        :param use_tqdm: parameter for `horovod` and `torch_distributed` backends. You can monitor
               training progress if use_tqdm=True.
        :param workers_per_node: parameter for `horovod` and `torch_distributed` backends. worker
               number on each node. default: 1.
        :param model_dir: parameter for `bigdl` backend. The path to save model. During the
               training, if checkpoint_trigger is defined and triggered, the model will be saved to
               model_dir.
        :param backend: You can choose "horovod",  "torch_distributed", "bigdl" or "spark" as
               backend. Default: `bigdl`.
        :return: an Estimator object.
        """

        if cluster is not None:
            assert cluster.name == "ray", "Only ray cluster is supported for now."
            assert backend in {"horovod", "torch_distributed"}, \
                       "Only horovod and torch_distributed are supported in ray cluster."


        if backend in {"horovod", "torch_distributed"}:
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
                                       backend=backend)
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
                                           workers_per_node=workers_per_node)
        else:
            raise ValueError("Only horovod, torch_distributed, bigdl and spark backends are "
                             f"supported for now, got backend: {backend}")
