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

import torch
import pytorch_lightning as pl


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
