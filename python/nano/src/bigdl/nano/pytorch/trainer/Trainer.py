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
import warnings
from logging import warning
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins.environments import LightningEnvironment
from torch import nn
from torch.fx.graph_module import GraphModule
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric

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
            pl_model = LightningModuleFromTorch(model, loss, optimizer, metrics)

        if onnx:
            try:
                from bigdl.nano.pytorch.runtime_binding.onnxrt_inference import\
                    bind_onnxrt_methods
                from bigdl.nano.pytorch.runtime_binding.base_inference import\
                    bind_base_inference_rt_methods
                return bind_onnxrt_methods(bind_base_inference_rt_methods(pl_model))
            except ImportError:
                raise RuntimeError("You should install onnx and onnxruntime to set `onnx=True`, "
                                   "or just set `onnx=False`.")
        else:
            return pl_model

    def quantize(self, pl_model: LightningModule,
                 calib_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 metric: Metric = None,
                 backend='inc',
                 conf='',
                 framework='pytorch_fx',
                 approach='static',
                 tuning_strategy='bayesian',
                 accuracy_criterion: dict = None,
                 timeout=0,
                 max_trials=1,
                 raw_return=False
                 ):
        """
        Calibrate a Pytorch-Lightning model for post-training quantization.

        :param pl_model:       A Pytorch-Lightning model to be quantized.
        :param calib_dataloader:    A torch.utils.data.dataloader.DataLoader object for calibration.
                                    Required for static quantization.
        :param val_dataloader:      A torch.utils.data.dataloader.DataLoader object for evaluation.
        :param metric:              A torchmetrics.metric.Metric object for evaluation.
        :param backend:             'inc' or 'nncf'('nncf' is not supported yet). Default: 'inc'.
        :param conf:        A path to conf yaml file for quantization.
                            Default: '', use default config.
        :param framework:   'pytorch', 'pytorch_fx', 'pytorch_ipex'. Default: 'pytorch_fx'.
                            Consistent with Intel Neural Compressor.
        :param approach:    'static' or 'dynamic'.
                            'static': post_training_static_quant,
                            'dynamic': post_training_dynamic_quant.
                            Default: 'static'.
        :param tuning_strategy:    'bayesian', 'basic', 'mse', 'sigopt'. Default: 'bayesian'.
        :param accuracy_criterion:  Tolerable accuracy drop.
                                    accuracy_criterion = {'relative': 0.1, higher_is_better: True}
                                    allows relative accuracy loss: 1%. accuracy_criterion =
                                    {'absolute': 0.99, higher_is_better:False} means accuracy < 0.99
                                     must be satisfied.
        :param timeout:     Tuning timeout (seconds). Default: 0,  which means early stop.
                            Combine with max_trials field to decide when to exit.
        :param max_trials:  Max tune times. Default: 1.
                            Combine with timeout field to decide when to exit.
                            "timeout=0, max_trials=1" means it will try quantization only once and
                            return satisfying best model.
        :param raw_return:  Decide which type to return. If set to True, a GraphModule will be
                            returned. If set to False, a pytorch lightning module will be returned.
        :return:            A GraphModule. If there is no model found, return None.
        """
        if backend == 'inc':
            from bigdl.nano.quantization import QuantizationINC
            from neural_compressor.experimental import common

            if approach not in ['static', 'dynamic']:
                raise ValueError("Approach should be 'static' or 'dynamic', "
                                 "{} is invalid.".format(approach))
            approach_map = {
                'static': 'post_training_static_quant',
                'dynamic': 'post_training_dynamic_quant'
            }
            approach = approach_map.get(approach)

            quantizer = QuantizationINC(framework=framework, conf=conf, approach=approach,
                                        tuning_strategy=tuning_strategy,
                                        accuracy_criterion=accuracy_criterion,
                                        timeout=timeout, max_trials=max_trials)
            if isinstance(pl_model, LightningModuleFromTorch):
                # LightningModuleFromTorch.forward fails to trace in FX, so replace it temporarily
                quantizer.model = pl_model.model
            else:
                quantizer.model = pl_model

            if val_dataloader and metric:
                from bigdl.nano.quantization.quantization_inc import TorchMetricForINC
                quantizer.eval_dataloader = val_dataloader
                quantizer.metric = common.Metric(TorchMetricForINC, metric=metric)
            if approach == 'post_training_static_quant':
                assert calib_dataloader, "calib_dataloader must not be None when approach is " \
                                         "post-training static quantization."
                quantizer.calib_dataloader = calib_dataloader
            quantized = quantizer()
            if quantized:
                if raw_return:
                    return quantized.model
                else:
                    from bigdl.nano.pytorch.runtime_binding.base_inference import\
                        bind_base_inference_rt_methods
                    from bigdl.nano.pytorch.runtime_binding.quantization_inference import\
                        bind_quantize_methods
                    return bind_quantize_methods(
                        bind_base_inference_rt_methods(pl_model), quantized.model)
            else:
                raise RuntimeError("Found no quantized model satisfying accuracy criterion.")
        else:
            raise NotImplementedError("Backend {} is not implemented.".format(backend))
