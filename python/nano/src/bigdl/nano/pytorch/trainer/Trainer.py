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
import copy
from logging import warning
from typing import Any, List, Optional, Union
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import _LRScheduler
from bigdl.nano.pytorch import InferenceOptimizer
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, TORCH_VERSION_LESS_1_11
from bigdl.nano.pytorch.utils import ChannelsLastCallback, save_model, load_model
from bigdl.nano.pytorch.algorithms import SelectiveBackprop
from bigdl.nano.pytorch.lightning import LightningModule
from bigdl.nano.pytorch.plugins.ddp_spawn import DDPSpawnPlugin
from bigdl.nano.pytorch.plugins.ddp_subprocess import DDPSubprocessPlugin
from bigdl.nano.deps.automl.hpo_api import create_hpo_searcher, check_hpo_status
from bigdl.nano.deps.ray.ray_api import distributed_ray
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.common import check_avx512
from bigdl.nano.utils import deprecated

distributed_backends = ["spawn", "ray", "subprocess"]


class Trainer(pl.Trainer):
    """
    Trainer for BigDL-Nano pytorch.

    This Trainer extends PyTorch Lightning Trainer by adding
    various options to accelerate pytorch training.
    """

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = False,
                 distributed_backend="subprocess",
                 cpu_for_each_process: Optional[List[List[int]]] = None,
                 use_hpo=False,
                 channels_last: bool = False,
                 auto_lr: Union[int, bool] = True,
                 precision: Union[str, int] = 32,
                 *args: Any, **kwargs: Any) -> None:
        """
        A pytorch lightning trainer that uses bigdl-nano optimization.

        :param num_processes: number of processes in distributed training. default: 4.
        :param use_ipex: whether we use ipex as accelerator for trainer. default: False.
        :param cpu_for_each_process: A list of length `num_processes`, each containing a list of
            indices of cpus each process will be using. default: None, and the cpu will be
            automatically and evenly distributed among processes.
        :param precision: Double precision (64), full precision (32), half precision (16)
            or bfloat16 precision (bf16), defaults to 32.
            Enable ipex bfloat16 weight prepack when `use_ipex=True` and `precision='bf16'`
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
                invalidInputError(False,
                                  f"The length of `cpu_for_each_process` ("
                                  f"{len(cpu_for_each_process)}) is not equal to the number of"
                                  f" processes {num_processes}.")

        accelerator = None

        if "algorithms" in kwargs:
            kwargs = self._add_algorithms(kwargs)

        if channels_last:
            callbacks = kwargs.get("callbacks")
            if callbacks:
                callbacks.append(ChannelsLastCallback())
            else:
                kwargs["callbacks"] = [ChannelsLastCallback()]

        self.use_ipex = use_ipex
        dtype = None
        if self.use_ipex and precision == 'bf16':
            # Enable ipex bfloat16 weight prepack and disable pytorch-lightning native AMP
            dtype = torch.bfloat16
            precision = 32

        # Confirm if cpu supports avx512
        if self.use_ipex and not check_avx512():
            if TORCH_VERSION_LESS_1_11:
                warning("Enable ipex<=1.11 in a cpu instruction set"
                        " without avx512 will crash."
                        "Fall back to regular pytorch.")
                self.use_ipex = False
            elif dtype == torch.bfloat16:
                warning("Enable IPEX bfloat16 in a cpu instruction set"
                        " without avx512 will crash. "
                        "Using 32-bit precision")
                dtype = None

        kwargs['precision'] = precision

        if num_processes == 1:
            from bigdl.nano.pytorch.strategies import create_IPEXStrategy
            strategy = create_IPEXStrategy(dtype=dtype) if self.use_ipex else None
            kwargs["strategy"] = strategy
            super().__init__(*args, **kwargs)
        else:
            plugin = None
            invalidInputError(distributed_backend in distributed_backends,
                              f"Distributed backends supported now are {distributed_backends},"
                              f" but get {distributed_backend}.")
            if "checkpoint_callback" in kwargs:
                if not kwargs["checkpoint_callback"]:
                    invalidInputError(False,
                                      f"`checkpoint_callback` set to False. "
                                      f"Currently, disable checkpoint callback make "
                                      f"distributed training backend work incorrect")
            if distributed_backend == "spawn":
                from bigdl.nano.pytorch.strategies import DDPSpawnStrategy
                strategy = DDPSpawnStrategy(num_processes=num_processes,
                                            cpu_for_each_process=cpu_for_each_process,
                                            use_ipex=self.use_ipex,
                                            dtype=dtype,
                                            auto_lr=auto_lr)
            elif distributed_backend == "subprocess":
                from bigdl.nano.pytorch.strategies import DDPSubprocessStrategy
                strategy = DDPSubprocessStrategy(num_processes=num_processes,
                                                 cpu_for_each_process=cpu_for_each_process,
                                                 use_ipex=self.use_ipex,
                                                 dtype=dtype,
                                                 auto_lr=auto_lr)
            elif distributed_backend == "ray":
                from bigdl.nano.pytorch.strategies import create_RayStrategy
                strategy = create_RayStrategy(num_workers=num_processes,
                                              use_ipex=self.use_ipex,
                                              dtype=dtype,
                                              auto_lr=auto_lr)
            kwargs["strategy"] = strategy
            super().__init__(*args, **kwargs)

        if use_hpo:
            self.hposearcher = create_hpo_searcher(trainer=self, num_processes=num_processes)
        else:
            self.hposearcher = None

    def _add_algorithms(self, kwargs):
        callbacks = kwargs.get("callbacks")
        for algorithm in kwargs['algorithms']:
            if isinstance(algorithm, SelectiveBackprop):
                if callbacks:
                    callbacks.append(algorithm)
                else:
                    kwargs["callbacks"] = [algorithm]
        del kwargs['algorithms']

        return kwargs

    @staticmethod
    def compile(model: nn.Module,
                loss: _Loss = None,
                optimizer: torch.optim.Optimizer = None,
                scheduler: _LRScheduler = None,
                metrics: List[Metric] = None):
        """
        Construct a pytorch-lightning model.

        If model is already a pytorch-lightning model,
        return model. If model is pytorch model, construct a new pytorch-lightning module
        with model, loss and optimizer.

        :param model:       A model instance.
        :param loss:        Loss to construct pytorch-lightning model.
                            Should be None if model is instance of pl.LightningModule.
        :param optimizer:   Optimizer to construct pytorch-lightning model Should be None.
                            if model is instance of pl.LightningModule.
        :param metrics:     A list of torchmetrics to validate/test performance.
        :return:            A LightningModule object.
        """
        invalidInputError(isinstance(model, nn.Module),
                          "Model must be instance of nn.Module but got {}".format(model.__class__))

        pl_model = None
        if isinstance(model, pl.LightningModule):
            invalidInputError(not (loss or optimizer),
                              "Loss and optimizer should be None if model"
                              " is a pytorch-lightning model.")
            pl_model = model
        else:
            pl_model = LightningModule(model, loss, optimizer, scheduler, metrics)

        return pl_model

    def search(self,
               model,
               resume: bool = False,
               target_metric=None,
               n_parallels=1,
               acceleration=False,
               input_sample=None,
               **kwargs):
        """
        Run HPO search. It will be called in Trainer.search().

        :param model: The model to be searched. It should be an auto model.
        :param resume: whether to resume the previous or start a new one,
            defaults to False.
        :param target_metric: the object metric to optimize,
            defaults to None.
        :param n_parallels: the number of parallel processes for running trials.
        :param acceleration: Whether to automatically consider the model after
            inference acceleration in the search process. It will only take
            effect if target_metric contains "latency". Default value is False.
        :param input_sample: A set of inputs for trace, defaults to None if you have
            trace before or model is a LightningModule with any dataloader attached.
        :return: the model with study meta info attached.
        """
        if not check_hpo_status(self.hposearcher):
            return None
        Trainer._log_api_event("search")

        return self.hposearcher.search(model,
                                       resume=resume,
                                       target_metric=target_metric,
                                       n_parallels=n_parallels,
                                       acceleration=acceleration,
                                       input_sample=input_sample,
                                       **kwargs)

    def search_summary(self):
        """
        Retrive a summary of trials.

        :return: A summary of all the trials. Currently the entire study is
            returned to allow more flexibility for further analysis and visualization.
        """
        if not check_hpo_status(self.hposearcher):
            return None
        return self.hposearcher.search_summary()

    @staticmethod
    @deprecated(func_name="bigdl.nano.pytorch.Trainer.trace",
                message="Please use `bigdl.nano.pytorch.InferenceOptimizer.trace` instead.")
    def trace(model: nn.Module,
              input_sample=None,
              accelerator: str = None,
              use_ipex: bool = False,
              thread_num: int = None,
              onnxruntime_session_options=None,
              logging: bool = True,
              **export_kwargs):
        """
        Trace a pytorch model and convert it into an accelerated module for inference.

        For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.

        :param model: An torch.nn.Module model, including pl.LightningModule.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached.
        :param accelerator: The accelerator to use, defaults to None meaning staying in Pytorch
                            backend. 'openvino', 'onnxruntime' and 'jit' are supported for now.
        :param use_ipex: whether we use ipex as accelerator for inferencing. default: False.
        :param thread_num: (optional) a int represents how many threads(cores) is needed for
                           inference, only valid for accelerator='onnxruntime'
                           or accelerator='openvino'.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. default: True.
        :param **kwargs: other extra advanced settings include
                         1. those be passed to torch.onnx.export function, only valid when
                         accelerator='onnxruntime'/'openvino', otherwise will be ignored.
                         2. if channels_last is set and use_ipex=True, we will transform the
                         data to be channels last according to the setting. Defaultly, channels_last
                         will be set to True if use_ipex=True.
        :return: Model with different acceleration.

        .. warning::
             ``bigdl.nano.pytorch.Trainer.trace`` will be deprecated in future release.

             Please use ``bigdl.nano.pytorch.InferenceOptimizer.trace`` instead.
        """
        return InferenceOptimizer.trace(model=model,
                                        input_sample=input_sample,
                                        accelerator=accelerator,
                                        use_ipex=use_ipex,
                                        thread_num=thread_num,
                                        onnxruntime_session_options=onnxruntime_session_options,
                                        logging=logging,
                                        **export_kwargs)

    @staticmethod
    @deprecated(func_name="bigdl.nano.pytorch.Trainer.quantize",
                message="Please use `bigdl.nano.pytorch.InferenceOptimizer.quantize` instead.")
    def quantize(model: nn.Module,
                 precision: str = 'int8',
                 accelerator: str = None,
                 use_ipex: bool = False,
                 calib_dataloader: DataLoader = None,
                 metric: Metric = None,
                 accuracy_criterion: dict = None,
                 approach: str = 'static',
                 method: str = None,
                 conf: str = None,
                 tuning_strategy: str = None,
                 timeout: int = None,
                 max_trials: int = None,
                 input_sample=None,
                 thread_num: int = None,
                 onnxruntime_session_options=None,
                 logging: bool = True,
                 **export_kwargs):
        """
        Calibrate a Pytorch-Lightning model for post-training quantization.

        :param model:           A model to be quantized. Model type should be an instance of
                                nn.Module.
        :param precision:       Global precision of quantized model,
                                supported type: 'int8', 'bf16', 'fp16', defaults to 'int8'.
        :param accelerator:     Use accelerator 'None', 'onnxruntime', 'openvino', defaults to None.
                                None means staying in pytorch.
        :param calib_dataloader:    A torch.utils.data.dataloader.DataLoader object for calibration.
                                    Required for static quantization.
                                    It's also used as validation dataloader.
        :param metric:              A torchmetrics.metric.Metric object for evaluation.
        :param accuracy_criterion:  Tolerable accuracy drop, defaults to None meaning no
                                    accuracy control.
                                    accuracy_criterion = {'relative': 0.1, 'higher_is_better': True}
                                    allows relative accuracy loss: 1%. accuracy_criterion =
                                    {'absolute': 0.99, 'higher_is_better':False} means accuracy
                                    must be smaller than 0.99.
        :param approach:    'static' or 'dynamic'.
                            'static': post_training_static_quant,
                            'dynamic': post_training_dynamic_quant.
                            Default: 'static'. OpenVINO supports static mode only.
        :param method:          Method to do quantization. When accelerator=None, supported
            methods: 'fx', 'eager', 'ipex', defaults to 'fx'. If you don't use ipex, suggest using
            'fx' which executes automatic optimizations like fusion. For more information, please
            refer to https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization.
            When accelerator='onnxruntime', supported methods: 'qlinear', 'integer', defaults
            to 'qlinear'. Suggest 'qlinear' for lower accuracy drop if using static quantization.
            More details in https://onnxruntime.ai/docs/performance/quantization.html.
            This argument doesn't take effect for OpenVINO, don't change it for OpenVINO.
        :param conf:        A path to conf yaml file for quantization.
                            Default: None, using default config.
        :param tuning_strategy:    'bayesian', 'basic', 'mse', 'sigopt'. Default: 'bayesian'.
        :param timeout:     Tuning timeout (seconds). Default: None,  which means early stop.
                            Combine with max_trials field to decide when to exit.
        :param max_trials:  Max tune times. Default: None, which means no tuning.
                            Combine with timeout field to decide when to exit.
                            "timeout=0, max_trials=1" means it will try quantization only once and
                            return satisfying best model.
        :param input_sample:      An input example to convert pytorch model into ONNX/OpenVINO.
        :param thread_num: (optional) a int represents how many threads(cores) is needed for
                           inference, only valid for accelerator='onnxruntime'
                           or accelerator='openvino'.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param logging: whether to log detailed information of model conversion, only valid when
                        accelerator='openvino', otherwise will be ignored. default: True.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        :return:            A accelerated Pytorch-Lightning Model if quantization is sucessful.

        .. warning::
             ``bigdl.nano.pytorch.Trainer.quantize`` will be deprecated in future release.

             Please use ``bigdl.nano.pytorch.InferenceOptimizer.quantize`` instead.
        """
        return InferenceOptimizer.quantize(model=model,
                                           precision=precision,
                                           accelerator=accelerator,
                                           use_ipex=use_ipex,
                                           calib_dataloader=calib_dataloader,
                                           metric=metric,
                                           accuracy_criterion=accuracy_criterion,
                                           approach=approach,
                                           method=method,
                                           conf=conf,
                                           tuning_strategy=tuning_strategy,
                                           timeout=timeout,
                                           max_trials=max_trials,
                                           input_sample=input_sample,
                                           thread_num=thread_num,
                                           onnxruntime_session_options=onnxruntime_session_options,
                                           logging=logging,
                                           **export_kwargs)

    @staticmethod
    def save(model: pl.LightningModule, path):
        """
        Save the model to local file.

        :param model: Any model of torch.nn.Module, including all models accelareted by
               Trainer.trace/Trainer.quantize.
        :param path: Path to saved model. Path should be a directory.
        """
        save_model(model, path)

    @staticmethod
    def load(path, model: pl.LightningModule = None):
        """
        Load a model from local.

        :param path: Path to model to be loaded. Path should be a directory.
        :param model: Required FP32 model to load pytorch model, it is needed if you accelerated
               the model with accelerator=None by Trainer.trace/Trainer.quantize. model
               should be set to None if you choose accelerator="onnxruntime"/"openvino"/"jit".
        :return: Model with different acceleration(None/OpenVINO/ONNX Runtime/JIT) or
                 precision(FP32/FP16/BF16/INT8).
        """
        return load_model(path, model)

    def save_checkpoint(    # type: ignore[override]
        self, filepath, weights_only: bool = False, storage_options: Optional[Any] = None
    ) -> None:
        """Save checkpoint after one train epoch."""
        # When using ipex==1.9 and custom lr_schedulers for training, if set `weights_only` to
        # False,`save_checkpoint` method will report an error of 'Unsupport storage type'
        # because the model is in 'xpu', so we temporarily move it to 'cpu',
        # then move it back after `save_checkpoint`.
        if self.use_ipex and TORCH_VERSION_LESS_1_10 and not weights_only:
            self.model.to('cpu')

        super().save_checkpoint(filepath, weights_only, storage_options)    # type: ignore
        if self.use_ipex and TORCH_VERSION_LESS_1_10 and not weights_only:
            self.model.to(self.strategy.root_device)    # type: ignore
