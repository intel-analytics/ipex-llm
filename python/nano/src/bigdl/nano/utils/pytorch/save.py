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

import yaml
from pathlib import Path
import copy

import torch
import pytorch_lightning as pl


def save_model(model: pl.LightningModule, path, compress_to_bf16=False):
    """
    Save the model to local file.

    :param model: Any model of torch.nn.Module, including all models accelareted by
            Trainer.trace/Trainer.quantize.
    :param path: Path to saved model. Path should be a directory.
    :param compress_to_bf16: Bool. This parameter only effective for jit, ipex or pure
            pytorch model with fp32 or bf16 precision.
    """
    path = Path(path)
    path.mkdir(parents=path.parent, exist_ok=True)
    if hasattr(model, '_save'):
        model._save(path, compress_to_bf16=compress_to_bf16)
    else:
        # typically for models of nn.Module, pl.LightningModule type
        meta_path = Path(path) / "nano_model_meta.yml"
        metadata = {
            'ModelType': 'PytorchModel',
            'checkpoint': 'saved_weight.pt',
            'compress_to_bf16': False
        }
        checkpoint_path = path / metadata['checkpoint']
        if compress_to_bf16:
            bf16_sd = transform_state_dict_to_dtype(model.state_dict(), dtype="bf16")
            torch.save(bf16_sd, checkpoint_path)
            metadata['compress_to_bf16'] = True
        else:
            torch.save(model.state_dict(), checkpoint_path)
        with open(meta_path, 'w+') as f:
            yaml.safe_dump(metadata, f)


def transform_state_dict_to_dtype(original_state_dict, dtype="bf16"):
    '''
    This function will transform a state dict to bf16
    '''
    # following block may fail
    sd_copy = copy.deepcopy(original_state_dict)
    for name in original_state_dict:
        if sd_copy[name].is_floating_point():
            if dtype == "bf16":
                sd_copy[name] = original_state_dict[name].bfloat16()
            if dtype == "fp32":
                sd_copy[name] = original_state_dict[name].float()
    return sd_copy
