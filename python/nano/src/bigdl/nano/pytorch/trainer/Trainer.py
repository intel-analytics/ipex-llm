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


from logging import warning
from typing import Any, List, Optional
import warnings

import pytorch_lightning as pl
import torch
from torchmetrics.metric import Metric
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch import nn
from torch.nn.modules.loss import _Loss

from bigdl.nano.common import check_avx512
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.plugins.ddp_spawn import DDPSpawnPlugin


distributed_backends = ["spawn", "ray"]


class Trainer(pl.Trainer):

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = False,
                 enable_bf16=False,
                 distributed_backend="spawn",
                 cpu_for_each_process: Optional[List[List[int]]] = None,
                 *args: Any, **kwargs: Any) -> None:
        """
        A pytorch lightning trainer that uses bigdl-nano optimization.
        :param num_processes: number of processes in distributed training. default: 4.
        :param use_ipex: whether we use ipex as accelerator for trainer. default: True.
        :param cpu_for_each_process: A list of length `num_processes`, each containing a list of
            indices of cpus each process will be using. default: None, and the cpu will be
            automatically and evenly distributed among processes.
        """

        # Check keyword arguments
        if "accelerator" in kwargs:
            warning(f"""Accelerator will be specified by bigdl-nano,
            accelerator entered {kwargs['accelerator']} will be ignored. """)

            kwargs.pop('accelerator')
        if "plugins" in kwargs:
            warning(f"""Plugins will be specified by bigdl-nano,
             plugines entered {kwargs['plugins']} will be ignored. """)

            kwargs.pop('plugins')
        if cpu_for_each_process is not None:
            if len(cpu_for_each_process) != num_processes:
                raise ValueError(f"The length of `cpu_for_each_process` ("
                                 f"{len(cpu_for_each_process)}) is not equal to the number of"
                                 f" processes {num_processes}.")

        # Initialize trainer
        if use_ipex and not check_avx512():
            warning("Enable ipex in a cpu instruction set"
                    " without avx512 may cause some random error."
                    "Fall back to cpu device.")
            use_ipex = False

        if num_processes == 1:
            accelerator = None
            if use_ipex:
                from bigdl.nano.pytorch.accelerators.ipex_accelerator import IPEXAccelerator
                accelerator = IPEXAccelerator(enable_bf16=enable_bf16)
            super().__init__(accelerator=accelerator, *args, **kwargs)
        else:
            plugin = None
            assert distributed_backend in distributed_backends, \
                f"Distributed backends supported now are spawn and ray," \
                " but get {distributed_backend}."
            if distributed_backend == "spawn":
                if use_ipex:
                    import intel_pytorch_extension as ipex
                    device = ipex.DEVICE
                else:
                    device = "cpu"
                plugin = DDPSpawnPlugin(parallel_devices=[
                    torch.device(device) for _ in range(num_processes)],
                    cpu_for_each_process=cpu_for_each_process,
                    cluster_environment=LightningEnvironment())
            elif distributed_backend == "ray":
                # Import RayPlugins may entangle with openmp even if it has not been used,
                # which leads to an unacceptably low performance.
                # So we import when we need.
                from bigdl.nano.pytorch.plugins.ray_distributed import RayPlugin
                plugin = RayPlugin(num_workers=num_processes,  # type: ignore
                                   use_ipex=use_ipex)

            accelerator = None
            if use_ipex:
                from bigdl.nano.pytorch.accelerators.ipex_accelerator import IPEXAccelerator
                accelerator = IPEXAccelerator(training_type_plugin=plugin,  # type: ignore
                                              enable_bf16=enable_bf16)

            super().__init__(accelerator=accelerator,
                             plugins=[plugin], *args, **kwargs)

    @staticmethod
    def compile(model: nn.Module,
                loss: _Loss = None,
                optimizer: torch.optim.Optimizer = None,
                metrics: List[Metric] = None,
                onnx: bool = False):
        """
        Construct a pytorch-lightning model. If model is already a pytorch-lightning model,
        return model. If model is pytorch model, construct a new pytorch-lightning module
        with model, loss and optimizer.

        :param model:       A model instance.
        :param loss:        Loss to construct pytorch-lightning model.
                            Should be None if model is instance of pl.LightningModule.
        :param optimizer:   Optimizer to construct pytorch-lightning model Should be None.
                            if model is instance of pl.LightningModule.
        :param metrics:     A list of torchmetrics to validate/test performance.
        :param onnx:        Indicates if onnxruntime support should be binded to the
                            returned model.
        :return:            A LightningModule object.
        """
        assert isinstance(model, nn.Module), \
            "Model must be instance of nn.Module but got {}".format(model.__class__)

        pl_model = None
        if isinstance(model, pl.LightningModule):
            assert not (loss or optimizer), \
                "Loss and optimizer should be None if model is a pytorch-lightning model."
            pl_model = model
        else:
            assert loss and optimizer, \
                "Loss and optimizer are required to construct a LightningModule instance."
            pl_model = LightningModuleFromTorch(model, loss, optimizer, metrics)

        if onnx:
            try:
                from bigdl.nano.pytorch.onnx.onnxrt_inference import bind_onnxrt_methods
                return bind_onnxrt_methods(pl_model)
            except ImportError:
                warnings.warn("You should install onnx and onnxruntime to set `onnx=True`")
                return pl_model
        else:
            return pl_model
