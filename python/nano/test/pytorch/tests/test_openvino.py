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
    def test_openvino(self):
        model = mobilenet_v3_small(num_classes=10)
        pl_model = Trainer.compile(model, openvino=True)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)

        pl_model.eval_openvino(x)
        assert pl_model.ir_up_to_date and pl_model.ir_model
        assert pl_model.forward != pl_model._torch_forward
        y = pl_model(x)
        assert y.shape == (10, 10)  

        pl_model.exit_openvino()
        assert pl_model.forward == pl_model._torch_forward

        pl_model.eval_openvino()
        assert pl_model.forward != pl_model._torch_forward
        pl_model.eval()
        assert pl_model.forward == pl_model._torch_forward

        # Test if correctly fall back to pytorch backend
        pl_model.eval_openvino()
        assert pl_model.forward != pl_model._torch_forward
        pl_model.train()
        assert pl_model.forward == pl_model._torch_forward  

    def test_openvino_inputsample_from_trainloader(self):
        trainer = Trainer(max_epochs=1)
        model = mobilenet_v3_small(num_classes=10)
        pl_model = Trainer.compile(model, loss=torch.nn.CrossEntropyLoss(),
                                   optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
                                   openvino=True)
        x = torch.rand((10, 3, 256, 256))
        y = torch.ones((10, ), dtype=torch.long)
        ds = TensorDataset(x, y)
        # TODO: batch size = 2 will cause error, need to fix this
        dataloader = DataLoader(ds, batch_size=10)
        trainer.fit(pl_model, dataloader)

        # Test if eval_openvino() and exit_openvino() work
        pl_model.eval_openvino()
        assert pl_model.ir_up_to_date and pl_model.ir_model
        assert pl_model.forward != pl_model._torch_forward
        y = pl_model(x)
        assert y.shape == (10, 10) 
        
        pl_model.exit_openvino()
        assert pl_model.forward == pl_model._torch_forward

        # Test if correctly fall back to training mode
        pl_model.eval_openvino()
        assert pl_model.forward != pl_model._torch_forward
        trainer.fit(pl_model, dataloader)
        assert pl_model.forward == pl_model._torch_forward and pl_model.ir_up_to_date == False
