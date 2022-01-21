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
import os
from unittest import TestCase

import pytest
import torchvision

from bigdl.nano.pytorch.trainer import Trainer
from pytorch.utils._train_torch_lightning import create_data_loader

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "../../data")


class TestLightningModuleFromTorch(TestCase):
    resnet18 = torchvision.models.resnet18()
    trainer = Trainer()

    def test_quantize_without_inc(self):
        pl_model = Trainer.compile(self.resnet18)
        dataloader = create_data_loader(data_dir, batch_size, num_workers, None)
        with pytest.raises(ImportError,
                           match="Intel Neural Compressor must be installed to use quantization."
                                 "Please install INC by: pip install neural-compressor."):
            self.trainer.quantize(pl_model, dataloader)
