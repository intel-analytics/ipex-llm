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
import subprocess
from unittest import TestCase
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from bigdl.nano.pytorch import Trainer
from torchvision.models.resnet import resnet18


class TestBF16(TestCase):
    def test_bf16_common(self):
        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        pl_model = Trainer.compile(model, loss, optimizer)
        try:
            bf16_model = trainer.quantize(pl_model, precision='bf16')
        except RuntimeError as e:
            possible_errors = [
                "Require torch>=1.10 and pytorch-lightning>=1.6.0."
            ]
            if str(e) in possible_errors:
                return
            else:
                raise e
        try:
            y_hat = bf16_model(x)
        except RuntimeError as e:
            possible_errors = [
                "Your machine or OS doesn't support BF16 instructions.",
                "BF16 ISA support is not enabled under current context."
            ]
            if str(e) in possible_errors:
                return
            else:
                raise e
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16

    def test_bf16_with_bf_isa_support(self):
        msg = subprocess.check_output(["lscpu"]).decode("utf-8")
        if not "avx512_core_bf16" in msg or "amx_bf16" in msg:
            print("BF16 instructions are not available in this machine. Exit testing...")
            return

        trainer = Trainer(max_epochs=1)
        model = resnet18(num_classes=10)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10,), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        pl_model = Trainer.compile(model, loss, optimizer)
        bf16_model = trainer.quantize(pl_model, precision='bf16')
        y_hat = bf16_model(x)
        assert y_hat.shape == (10, 10) and y_hat.dtype == torch.bfloat16


if __name__ == '__main__':
    pytest.main([__file__])
