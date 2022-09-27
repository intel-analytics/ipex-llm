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
from unittest import TestCase
from torchmetrics import F1
from bigdl.nano.pytorch.trainer import Trainer
from torchvision.models.mobilenetv3 import mobilenet_v3_small
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data.dataloader import DataLoader


class TestOpenVINO(TestCase):
    def test_trainer_quantize_openvino(self):
        trainer = Trainer()
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        # Case1: Trace and quantize
        openvino_model = trainer.trace(model, accelerator='openvino', input_sample=x)
        optimized_model = trainer.quantize(openvino_model, accelerator='openvino',
                                           calib_dataloader=dataloader)
        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        # Case2: Quantize directly from pytorch
        optimized_model = trainer.quantize(model, accelerator='openvino',
                                           calib_dataloader=dataloader)

        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        trainer.validate(optimized_model, dataloader)
        trainer.test(optimized_model, dataloader)
        trainer.predict(optimized_model, dataloader)

    def test_trainer_quantize_openvino_with_tuning(self):
        trainer = Trainer()
        model = mobilenet_v3_small(num_classes=10)

        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        ds = TensorDataset(x, y)
        dataloader = DataLoader(ds, batch_size=2)

        optimized_model = trainer.quantize(model, accelerator='openvino',
                                           calib_dataloader=dataloader,
                                           metric=F1(10))

        y_hat = optimized_model(x[0:3])
        assert y_hat.shape == (3, 10)
        y_hat = optimized_model(x)
        assert y_hat.shape == (10, 10)

        trainer.validate(optimized_model, dataloader)
        trainer.test(optimized_model, dataloader)
        trainer.predict(optimized_model, dataloader)