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
from pathlib import Path
from typing import Any, List, Optional
import pytorch_lightning as pl
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.utils.data import DataLoader
from torchmetrics.metric import Metric
from torch.optim.lr_scheduler import _LRScheduler
import yaml
from bigdl.nano.pytorch.utils import TORCH_VERSION_LESS_1_10, TORCH_VERSION_LESS_1_11, \
    LIGHTNING_VERSION_LESS_1_6
from bigdl.nano.pytorch.utils import ChannelsLastCallback
from bigdl.nano.pytorch.amp import BF16Model
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.plugins.ddp_spawn import DDPSpawnPlugin
from bigdl.nano.pytorch.plugins.ddp_subprocess import DDPSubprocessPlugin

from bigdl.nano.deps.automl.hpo_api import create_hpo_searcher, check_hpo_status
from bigdl.nano.deps.ray.ray_api import distributed_ray
from bigdl.nano.deps.ipex.ipex_api import create_IPEXAccelerator, create_IPEXAccelerator_1_9
from bigdl.nano.openvino import PytorchOpenVINOModel, load_openvino_model
from bigdl.nano.deps.onnxruntime.onnxruntime_api import PytorchONNXRuntimeModel, \
    load_onnxruntime_model
from bigdl.nano.deps.neural_compressor.inc_api import load_inc_model, quantize as inc_quantize
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
from bigdl.nano.common import check_avx512
distributed_backends = ["spawn", "ray", "subprocess"]


class Trainer(pl.Trainer):
    """
    Trainer for BigDL-Nano pytorch.

    This Trainer extends PyTorch Lightning Trainer by adding
    various options to accelerate pytorch training.
    """

    def __init__(self, num_processes: int = 1,
                 use_ipex: bool = False,
                 enable_bf16=False,
                 distributed_backend="subprocess",
                 cpu_for_each_process: Optional[List[List[int]]] = None,
                 use_hpo=False,
                 channels_last: bool = False,
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
                invalidInputError(False,
                                  f"The length of `cpu_for_each_process` ("
                                  f"{len(cpu_for_each_process)}) is not equal to the number of"
                                  f" processes {num_processes}.")

        accelerator = None

        if channels_last:
            callbacks = kwargs.get("callbacks")
            if callbacks:
                callbacks.append(ChannelsLastCallback())
            else:
                kwargs["callbacks"] = [ChannelsLastCallback()]

        if TORCH_VERSION_LESS_1_11 and use_ipex and not check_avx512():
            warning("Enable ipex<=1.10 in a cpu instruction set"
                    " without avx512 will crash."
                    "Fall back to regular pytorch.")
            use_ipex = False

        self.use_ipex = use_ipex

        if num_processes == 1:
            if LIGHTNING_VERSION_LESS_1_6:
                if self.use_ipex:
                    if TORCH_VERSION_LESS_1_10:
                        accelerator = create_IPEXAccelerator_1_9(enable_bf16=enable_bf16)
                    else:
                        accelerator = create_IPEXAccelerator(enable_bf16=enable_bf16)
                super().__init__(accelerator=accelerator, *args, **kwargs)  # type: ignore
            else:
                from bigdl.nano.pytorch.strategies import create_IPEXStrategy
                strategy = create_IPEXStrategy(enable_bf16=enable_bf16) if self.use_ipex else None
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
            if LIGHTNING_VERSION_LESS_1_6:
                if distributed_backend == "spawn":
                    plugin = DDPSpawnPlugin(num_processes=num_processes,
                                            cpu_for_each_process=cpu_for_each_process,
                                            use_ipex=self.use_ipex,
                                            enable_bf16=enable_bf16)
                elif distributed_backend == "subprocess":
                    plugin = DDPSubprocessPlugin(num_processes=num_processes,
                                                 cpu_for_each_process=cpu_for_each_process,
                                                 use_ipex=self.use_ipex,
                                                 enable_bf16=enable_bf16)
                elif distributed_backend == "ray":
                    # Import RayPlugins may entangle with openmp even if it has not been used,
                    # which leads to an unacceptably low performance.
                    # So we import when we need.
                    plugin = distributed_ray(num_workers=num_processes,  # type: ignore
                                             use_ipex=self.use_ipex,
                                             enable_bf16=enable_bf16)
                if self.use_ipex and TORCH_VERSION_LESS_1_10:
                    accelerator = create_IPEXAccelerator_1_9(training_type_plugin=plugin,
                                                             enable_bf16=enable_bf16)
                super().__init__(accelerator=accelerator, plugins=[plugin],  # type: ignore
                                 *args, **kwargs)
            else:
                if distributed_backend == "spawn":
                    from bigdl.nano.pytorch.strategies import DDPSpawnStrategy
                    strategy = DDPSpawnStrategy(num_processes=num_processes,
                                                cpu_for_each_process=cpu_for_each_process,
                                                use_ipex=self.use_ipex,
                                                enable_bf16=enable_bf16)
                elif distributed_backend == "subprocess":
                    from bigdl.nano.pytorch.strategies import DDPSubprocessStrategy
                    strategy = DDPSubprocessStrategy(num_processes=num_processes,
                                                     cpu_for_each_process=cpu_for_each_process,
                                                     use_ipex=self.use_ipex,
                                                     enable_bf16=enable_bf16)
                elif distributed_backend == "ray":
                    from bigdl.nano.pytorch.strategies import create_RayStrategy
                    strategy = create_RayStrategy(num_workers=num_processes,
                                                  use_ipex=self.use_ipex,
                                                  enable_bf16=enable_bf16)
                kwargs["strategy"] = strategy
                super().__init__(*args, **kwargs)

        if use_hpo:
            self.hposearcher = create_hpo_searcher(trainer=self, num_processes=num_processes)
        else:
            self.hposearcher = None

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
            pl_model = LightningModuleFromTorch(model, loss, optimizer, scheduler, metrics)

        return pl_model

    def search(self,
               model,
               resume: bool = False,
               target_metric=None,
               n_parallels=1,
               **kwargs):
        """
        Run HPO search. It will be called in Trainer.search().

        :param model: The model to be searched. It should be an auto model.
        :param resume: whether to resume the previous or start a new one,
            defaults to False.
        :param target_metric: the object metric to optimize,
            defaults to None.
        :param n_parallels: the number of parallel processes for running trials.
        :return: the model with study meta info attached.
        """
        if not check_hpo_status(self.hposearcher):
            return None
        Trainer._log_api_event("search")

        return self.hposearcher.search(model,
                                       resume=resume,
                                       target_metric=target_metric,
                                       n_parallels=n_parallels,
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
    def quantize(model,  # remove the type requirement for type checking
                 precision: str = 'int8',
                 accelerator=None,
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
                 onnxruntime_session_options=None,
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
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        :return:            A accelerated Pytorch-Lightning Model if quantization is sucessful.
        """
        if precision == 'bf16':
            if accelerator is None:
                bf16_model = BF16Model(model)
                return bf16_model
            else:
                invalidInputError(False,
                                  "Accelerator {} is invalid for BF16.".format(accelerator))
        if precision == 'int8':
            if not accelerator or accelerator == 'onnxruntime':
                method_map = {
                    None: {
                        'fx': 'pytorch_fx',
                        'eager': 'pytorch',
                        'ipex': 'pytorch_ipex',
                        None: 'pytorch_fx'  # default
                    },
                    'onnxruntime': {
                        'qlinear': 'onnxrt_qlinearops',
                        'integer': 'onnxrt_integerops',
                        None: 'onnxrt_qlinearops'  # default
                    }
                }
                framework = method_map[accelerator].get(method, None)
                if accelerator == "onnxruntime":
                    if not type(model).__name__ == 'PytorchONNXRuntimeModel':
                        # try to establish onnx model
                        if input_sample is None:
                            # input_sample can be a dataloader
                            input_sample = calib_dataloader
                        model = Trainer.trace(
                            model,
                            input_sample=input_sample,
                            accelerator='onnxruntime',
                            onnxruntime_session_options=onnxruntime_session_options,
                            **export_kwargs)
                """
                If accelerator==None, quantized model returned should be an object of PytorchModel
                which is defined by neural-compressor containing a `GraphModule` for inference.
                Otherwise accelerator=='onnxruntime', it returns an ONNXModel object. A supported
                model which is able to run on Pytorch or ONNXRuntime can be fetched by
                `quantized_model.model`.
                """
                return inc_quantize(model, calib_dataloader, metric,
                                    framework=framework,
                                    conf=conf,
                                    approach=approach,
                                    tuning_strategy=tuning_strategy,
                                    accuracy_criterion=accuracy_criterion,
                                    timeout=timeout,
                                    max_trials=max_trials)

            elif accelerator == 'openvino':
                model_type = type(model).__name__
                if not model_type == 'PytorchOpenVINOModel':
                    if input_sample is None:
                        # input_sample can be a dataloader
                        input_sample = calib_dataloader
                    model = Trainer.trace(model,
                                          input_sample=input_sample,
                                          accelerator='openvino',
                                          **export_kwargs)
                invalidInputError(type(model).__name__ == 'PytorchOpenVINOModel',
                                  "Invalid model to quantize. Please use a nn.Module or a model "
                                  "from trainer.trance(accelerator=='openvino')")
                drop_type = None
                higher_is_better = None
                maximal_drop = None
                if metric:
                    if not isinstance(accuracy_criterion, dict):
                        accuracy_criterion = {'relative': 0.99, 'higher_is_better': True}

                    drop_type = 'relative' if 'relative' in accuracy_criterion else 'absolute'
                    higher_is_better = accuracy_criterion.get('higher_is_better', None)
                    maximal_drop = accuracy_criterion.get(drop_type, None)

                kwargs = {
                    "metric": metric,
                    "higher_better": higher_is_better,
                    "drop_type": drop_type,
                    "maximal_drop": maximal_drop,
                    "max_iter_num": max_trials,
                    # TODO following two keys are optional, if there is need, we can add them
                    # "n_requests": None,
                    # "sample_size": 300
                }
                return model.pot(calib_dataloader, **kwargs)
            else:
                invalidInputError(False,
                                  "Accelerator {} is invalid.".format(accelerator))
        invalidInputError(False,
                          "Precision {} is invalid.".format(precision))

    @staticmethod
    def trace(model: nn.Module,
              input_sample=None,
              accelerator=None,
              onnxruntime_session_options=None,
              **export_kwargs):
        """
        Trace a pytorch model and convert it into an accelerated module for inference.

        For example, this function returns a PytorchOpenVINOModel when accelerator=='openvino'.

        :param model: An torch.nn.Module model, including pl.LightningModule.
        :param input_sample: A set of inputs for trace, defaults to None if you have trace before or
                             model is a LightningModule with any dataloader attached.
        :param accelerator: The accelerator to use, defaults to None meaning staying in Pytorch
                            backend. 'openvino' and 'onnxruntime' are supported for now.
        :param onnxruntime_session_options: The session option for onnxruntime, only valid when
                                            accelerator='onnxruntime', otherwise will be ignored.
        :param **export_kwargs: will be passed to torch.onnx.export function.
        :return: Model with different acceleration(OpenVINO/ONNX Runtime).
        """
        invalidInputError(
            isinstance(model, nn.Module) and not isinstance(model, AcceleratedLightningModule),
            "Expect a nn.Module instance that is not traced or quantized"
            "but got type {}".format(type(model))
        )
        if accelerator == 'openvino':
            return PytorchOpenVINOModel(model, input_sample, **export_kwargs)
        if accelerator == 'onnxruntime':
            return PytorchONNXRuntimeModel(model, input_sample, onnxruntime_session_options,
                                           **export_kwargs)
        invalidInputError(False, "Accelerator {} is invalid.".format(accelerator))

    @staticmethod
    def save(model: LightningModule, path):
        """
        Save the model to local file.

        :param model: Any model of torch.nn.Module, including PytorchOpenVINOModel,
         PytorchONNXModel.
        :param path: Path to saved model. Path should be a directory.
        """
        path = Path(path)
        Path.mkdir(path, exist_ok=True)
        if hasattr(model, '_save'):
            model._save(path)
        else:
            # typically for models of nn.Module, LightningModule and LightningModuleFromTorch type
            meta_path = Path(path) / "nano_model_meta.yml"
            with open(meta_path, 'w+') as f:
                metadata = {
                    'ModelType': 'PytorchModel',
                    'checkpoint': 'saved_weight.pt'
                }
                yaml.safe_dump(metadata, f)
            checkpoint_path = path / metadata['checkpoint']
            torch.save(model.state_dict(), checkpoint_path)

    @staticmethod
    def load(path, model: LightningModule = None):
        """
        Load a model from local.

        :param path: Path to model to be loaded. Path should be a directory.
        :param model: Required FP32 model to load pytorch model. Optional for ONNX/OpenVINO.
        :return: Model with different acceleration(None/OpenVINO/ONNX Runtime) or
                 precision(FP32/FP16/BF16/INT8).
        """
        path = Path(path)
        if not path.exists():
            invalidInputError(False, "{} doesn't exist.".format(path))
        meta_path = path / "nano_model_meta.yml"
        if not meta_path.exists():
            invalidInputError(False, "File {} is required to load model.".format(str(meta_path)))
        with open(meta_path, 'r') as f:
            metadata = yaml.safe_load(f)
        model_type = metadata.get('ModelType', None)
        if model_type == 'PytorchOpenVINOModel':
            invalidInputError(model is None,
                              "Argument 'model' must be None for OpenVINO loading.")
            return load_openvino_model(path)
        if model_type == 'PytorchONNXRuntimeModel':
            invalidInputError(model is None,
                              "Argument 'model' must be None for ONNX Runtime loading.")
            return load_onnxruntime_model(path)
        if model_type == 'PytorchQuantizedModel':
            return load_inc_model(path, model, 'pytorch')
        if isinstance(model, nn.Module):
            # typically for models of nn.Module, LightningModule and LightningModuleFromTorch type
            model = copy.deepcopy(model)
            checkpoint_path = metadata.get('checkpoint', None)
            if checkpoint_path:
                checkpoint_path = path / metadata['checkpoint']
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict)
                return model
            else:
                invalidInputError(False, "Key 'checkpoint' must be specified.")
        else:
            invalidInputError(False,
                              "ModelType {} or argument 'model={}' is not acceptable for pytorch"
                              " loading.".format(model_type, type(model)))

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

        if LIGHTNING_VERSION_LESS_1_6:
            super().save_checkpoint(filepath, weights_only)
        else:
            super().save_checkpoint(filepath, weights_only, storage_options)    # type: ignore
        if self.use_ipex and TORCH_VERSION_LESS_1_10 and not weights_only:
            if LIGHTNING_VERSION_LESS_1_6:
                self.model.to(self.training_type_plugin.root_device)
            else:
                self.model.to(self.strategy.root_device)    # type: ignore
