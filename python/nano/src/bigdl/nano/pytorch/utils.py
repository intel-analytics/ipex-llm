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

import operator
from pytorch_lightning.utilities.imports import _compare_version
from types import MethodType
import pytorch_lightning as pl
from typing import Optional
import torch
import torch.nn as nn
import copy
import yaml
from logging import warning
from bigdl.nano.deps.openvino.openvino_api import load_openvino_model
from bigdl.nano.deps.ipex.ipex_api import load_ipexjit_model
from bigdl.nano.deps.onnxruntime.onnxruntime_api import load_onnxruntime_model
from bigdl.nano.deps.neural_compressor.inc_api import load_inc_model
from bigdl.nano.utils.log4Error import invalidInputError
from pathlib import Path

TORCH_VERSION_LESS_1_10 = _compare_version("torch", operator.lt, "1.10")
TORCH_VERSION_LESS_1_11 = _compare_version("torch", operator.lt, "1.11")
TORCH_VERSION_LESS_1_12 = _compare_version("torch", operator.lt, "1.12")
TORCHVISION_VERSION_LESS_1_12 = _compare_version("torchvision", operator.lt, "0.12.0")


def batch_call(func):
    """
    Decorator to extending hook of pl_module.

    Extending behavior hook on_before_batch_transfer to convert data to channels_last
    for each batch.
    """

    def on_before_batch_transfer(self, batch, dataloader_idx):

        def convert_channels_last(batch):
            if isinstance(batch, torch.Tensor) and batch.dim() == 4:
                batch = batch.to(memory_format=torch.channels_last)
            elif isinstance(batch, list) or isinstance(batch, tuple):
                batch = list(batch)
                for index, t in enumerate(batch):
                    batch[index] = convert_channels_last(t)
            return batch
        batch = func(batch, dataloader_idx)
        batch = convert_channels_last(batch)
        return batch
    return on_before_batch_transfer


class ChannelsLastCallback(pl.Callback):
    """Custom pl.Callback for converting model and data to channels_last."""

    def setup(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        """Override hook setup to convert model to channels_last and wrap DataHook."""
        # TODO: Add check for module_states
        try:
            pl_module = pl_module.to(memory_format=torch.channels_last)
        except Exception as e:
            warning(f"Convert model to channels last failed, \
                    fall back to origin memory format. Exception msg: {e}")
            return super().setup(trainer, pl_module, stage)
        fn_old = getattr(pl_module, "on_before_batch_transfer")
        fn = batch_call(fn_old)
        setattr(pl_module, "on_before_batch_transfer_origin", fn_old)
        pl_module.on_before_batch_transfer = MethodType(fn, pl_module)
        return super().setup(trainer, pl_module, stage)

    def teardown(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        """Undo the changes to pl_module at end of fit, validate, tests, or predict."""
        if hasattr(pl_module, "on_before_batch_transfer_origin"):
            setattr(pl_module, "on_before_batch_transfer",
                    pl_module.on_before_batch_transfer_origin)
            delattr(pl_module, "on_before_batch_transfer_origin")
        return super().teardown(trainer, pl_module, stage)


def save_model(model: pl.LightningModule, path):
    """
    Save the model to local file.

    :param model: Any model of torch.nn.Module, including all models accelareted by
            Trainer.trace/Trainer.quantize.
    :param path: Path to saved model. Path should be a directory.
    """
    path = Path(path)
    path.mkdir(parents=path.parent, exist_ok=True)
    if hasattr(model, '_save'):
        model._save(path)
    else:
        # typically for models of nn.Module, pl.LightningModule type
        meta_path = Path(path) / "nano_model_meta.yml"
        with open(meta_path, 'w+') as f:
            metadata = {
                'ModelType': 'PytorchModel',
                'checkpoint': 'saved_weight.pt'
            }
            yaml.safe_dump(metadata, f)
        checkpoint_path = path / metadata['checkpoint']
        torch.save(model.state_dict(), checkpoint_path)


def load_model(path, model: pl.LightningModule = None):
    """
    Load a model from local.

    :param path: Path to model to be loaded. Path should be a directory.
    :param model: Required FP32 model to load pytorch model, it is needed if you accelerated
            the model with accelerator=None by Trainer.trace/Trainer.quantize. model
            should be set to None if you choose accelerator="onnxruntime"/"openvino"/"jit".
    :return: Model with different acceleration(None/OpenVINO/ONNX Runtime/JIT) or
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
    if model_type == 'PytorchIPEXJITModel':
        return load_ipexjit_model(path, model)
    if isinstance(model, nn.Module):
        # typically for models of nn.Module, pl.LightningModule type
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
