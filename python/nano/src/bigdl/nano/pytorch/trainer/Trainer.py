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
from tempfile import TemporaryDirectory
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
from torch.optim.lr_scheduler import _LRScheduler
import yaml
from bigdl.nano.common import check_avx512
from bigdl.nano.pytorch.lightning import LightningModuleFromTorch
from bigdl.nano.pytorch.plugins.ddp_spawn import DDPSpawnPlugin
from bigdl.nano.deps.automl.hpo_api import create_hpo_searcher, check_hpo_status
from bigdl.nano.deps.ray.ray_api import distributed_ray
from bigdl.nano.deps.ipex.ipex_api import create_IPEXAccelerator, ipex_device
from bigdl.nano.deps.openvino.openvino_api import PytorchOpenVINOModel, load_openvino_model
from bigdl.nano.deps.onnxruntime.onnxruntime_api import bind_onnxrt_methods,\
    PytorchONNXRuntimeModel, load_onnxruntime_model
from bigdl.nano.deps.neural_compressor.inc_api import QuantizationINC, PytorchQuantizedModel,\
    check_pytorch_dataloaders, load_inc_model
from bigdl.nano.utils.log4Error import invalidInputError
from bigdl.nano.utils.inference.pytorch.model import AcceleratedLightningModule
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

        if use_hpo:
            self.hposearcher = create_hpo_searcher(trainer=self)
        else:
            self.hposearcher = None

        # Initialize trainer
        if use_ipex and not check_avx512():
            warning("Enable ipex in a cpu instruction set"
                    " without avx512 may cause some random error."
                    "Fall back to cpu device.")
            use_ipex = False

        if num_processes == 1:
            accelerator = None
            if use_ipex:
                accelerator = create_IPEXAccelerator(enable_bf16=enable_bf16)
            super().__init__(accelerator=accelerator, *args, **kwargs)
        else:
            plugin = None
            invalidInputError(distributed_backend in distributed_backends,
                              f"Distributed backends supported now are subprocess, spawn and ray,"
                              f" but get {distributed_backend}.")
            if distributed_backend == "spawn":
                if use_ipex:
                    device = ipex_device()
                else:
                    device = "cpu"
                plugin = DDPSpawnPlugin(parallel_devices=[
                    torch.device(device) for _ in range(num_processes)],
                    cpu_for_each_process=cpu_for_each_process,
                    cluster_environment=LightningEnvironment())
            elif distributed_backend == "subprocess":
                from bigdl.nano.pytorch.plugins.ddp_subprocess import DDPSubprocessPlugin
                if use_ipex:
                    import intel_pytorch_extension as ipex
                    device = ipex.DEVICE
                else:
                    device = "cpu"
                plugin = DDPSubprocessPlugin(parallel_devices=[
                    torch.device(device) for _ in range(num_processes)],
                    cpu_for_each_process=cpu_for_each_process,
                    cluster_environment=LightningEnvironment())
            elif distributed_backend == "ray":
                # Import RayPlugins may entangle with openmp even if it has not been used,
                # which leads to an unacceptably low performance.
                # So we import when we need.
                plugin = distributed_ray(num_workers=num_processes,  # type: ignore
                                         use_ipex=use_ipex,
                                         device=ipex_device())

            accelerator = None
            if use_ipex:
                accelerator = create_IPEXAccelerator(training_type_plugin=plugin,  # type: ignore
                                                     enable_bf16=enable_bf16)

            super().__init__(accelerator=accelerator,
                             plugins=[plugin], *args, **kwargs)

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
               **kwargs):
        """
        Run HPO search. It will be called in Trainer.search().

        :param model: The model to be searched. It should be an auto model.
        :param resume: whether to resume the previous or start a new one,
            defaults to False.
        :param target_metric: the object metric to optimize,
            defaults to None.
        :param return: the model with study meta info attached.
        """
        if not check_hpo_status(self.hposearcher):
            return None
        Trainer._log_api_event("search")

        return self.hposearcher.search(model,
                                       resume=resume,
                                       target_metric=target_metric,
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

    def quantize(self, model,  # remove the type requirement for type checking
                 precision='int8',
                 accelerator=None,
                 calib_dataloader: DataLoader = None,
                 val_dataloader: DataLoader = None,
                 metric: Optional[Metric] = None,
                 accuracy_criterion: dict = {'relative': 0.99, 'higher_is_better': True},
                 approach='static',
                 method='fx',
                 conf: Optional[str] = None,
                 tuning_strategy='bayesian',
                 timeout=0,
                 max_trials=1,
                 input_sample=None
                 ):
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
        :param val_dataloader:      A torch.utils.data.dataloader.DataLoader object for evaluation.
        :param metric:              A torchmetrics.metric.Metric object for evaluation.
        :param accuracy_criterion:  Tolerable accuracy drop.
                                    accuracy_criterion = {'relative': 0.1, 'higher_is_better': True}
                                    allows relative accuracy loss: 1%. accuracy_criterion =
                                    {'absolute': 0.99, 'higher_is_better':False} means accuracy
                                    must be smaller than 0.99.
        :param approach:    'static' or 'dynamic'.
                            'static': post_training_static_quant,
                            'dynamic': post_training_dynamic_quant.
                            Default: 'static'.
        :param method:          Method to do quantization. When accelerator=None, supported
            methods: 'fx', 'eager', 'ipex', defaults to 'fx'. If you don't use ipex, suggest using
            'fx' which executes automatic optimizations like fusion. For more information, please
            refer to https://pytorch.org/docs/stable/quantization.html#eager-mode-quantization.
            When accelerator='onnxruntime', supported methods: 'qlinear', 'integer', defaults
            to 'qlinear'. Suggest 'qlinear' for lower accuracy drop if using static quantization.
            More details in https://onnxruntime.ai/docs/performance/quantization.html.
        :param conf:        A path to conf yaml file for quantization.
                            Default: None, using default config.
        :param tuning_strategy:    'bayesian', 'basic', 'mse', 'sigopt'. Default: 'bayesian'.
        :param timeout:     Tuning timeout (seconds). Default: 0,  which means early stop.
                            Combine with max_trials field to decide when to exit.
        :param max_trials:  Max tune times. Default: 1.
                            Combine with timeout field to decide when to exit.
                            "timeout=0, max_trials=1" means it will try quantization only once and
                            return satisfying best model.
        :input_sample:      An input example to convert pytorch model into ONNX/OpenVINO.

        :return:            A accelerated Pytorch-Lightning Model if quantization is sucessful.
        """
        if not accelerator or accelerator == 'onnxruntime':
            # check if dataloader is of legal format
            check_pytorch_dataloaders(model, [calib_dataloader, val_dataloader])

            if approach not in ['static', 'dynamic']:
                invalidInputError(False,
                                  "Approach should be 'static' or 'dynamic', "
                                  "{} is invalid.".format(approach))
            approach_map = {
                'static': 'post_training_static_quant',
                'dynamic': 'post_training_dynamic_quant'
            }
            approach = approach_map.get(approach)
            if accelerator is None:
                framework = 'pytorch_{}'.format(method)
            if accelerator == 'onnxruntime':
                framework = "{}_{}ops".format('onnxrt', method)
            quantizer = QuantizationINC(framework=framework, conf=conf, approach=approach,
                                        tuning_strategy=tuning_strategy,
                                        accuracy_criterion=accuracy_criterion,
                                        timeout=timeout, max_trials=max_trials)
            if accelerator == "onnxruntime":
                if not type(model).__name__ == 'PytorchONNXRuntimeModel':
                    # try to establish onnx model
                    if not input_sample:
                        # input_sample can be a dataloader
                        input_sample = calib_dataloader or val_dataloader
                    model = Trainer.trace(model,
                                          input_sample=input_sample,
                                          accelerator='onnxruntime')
                model = model.onnx_model
            """
            If accelerator==None, quantized model returned should be an object of PytorchModel
            which is defined by neural-compressor containing a `GraphModule` for inference.
            Otherwise accelerator=='onnxruntime', it returns an ONNXModel object. A supported
            model which is able to run on Pytorch or ONNXRuntime can be fetched by
            `quantized_model.model`.
            """
            quantized_model = quantizer.post_training_quantize(model, calib_dataloader,
                                                               val_dataloader, metric)
            if accelerator is None:
                return PytorchQuantizedModel(quantized_model)
            elif accelerator == 'onnxruntime':
                with TemporaryDirectory() as dir:
                    saved_onnx = Path(dir) / 'tmp.onnx'
                    quantized_model.save(saved_onnx)
                    return PytorchONNXRuntimeModel(str(saved_onnx))
        elif accelerator == 'openvino':
            model_type = type(model).__name__
            if not model_type == 'PytorchOpenVINOModel':
                if not input_sample:
                    # input_sample can be a dataloader
                    input_sample = calib_dataloader
                model = Trainer.trace(model,
                                      input_sample=input_sample,
                                      accelerator='openvino')
            invalidInputError(type(model).__name__ == 'PytorchOpenVINOModel',
                              "Invalid model to quantize. Please use a nn.Module or a model "
                              "from trainer.trance(accelerator=='openvino')")
            drop_type = 'relative' if 'relative' in accuracy_criterion else 'absolute'
            kwargs = {
                "metric": metric,
                "higher_better": accuracy_criterion['higher_is_better'],
                "drop_type": drop_type,
                "maximal_drop": accuracy_criterion[drop_type],
                "max_iter_num": max_trials,
                # TODO following two keys are optional, if there is need, we can add them
                # "n_requests": None,
                # "sample_size": 300
            }
            return model.pot(calib_dataloader, **kwargs)
        else:
            invalidInputError(False,
                              "Accelerator {} is invalid.".format(accelerator))

    @staticmethod
    def trace(model: nn.Module,
              input_sample=None,
              accelerator=None,
              onnxruntime_session_options=None):
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
        :return: Model with different acceleration(OpenVINO/ONNX Runtime).
        """
        invalidInputError(
            isinstance(model, nn.Module) and not isinstance(model, AcceleratedLightningModule),
            "Expect a nn.Module instance that is not traced or quantized"
            "but got type {}".format(type(model))
        )
        if accelerator == 'openvino':
            return PytorchOpenVINOModel(model, input_sample)
        if accelerator == 'onnxruntime':
            return PytorchONNXRuntimeModel(model, input_sample, onnxruntime_session_options)
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
