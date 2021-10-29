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

import torch
from pytorch_lightning import LightningModule
from torch.nn.modules.loss import _Loss

from _train_torch_lightning import create_data_loader, data_transform
from bigdl.nano.pytorch.onnxruntime_support import onnxruntime
from bigdl.nano.pytorch.vision.models import vision


def to_lightning(loss: _Loss, optimizer: torch.optim, **opt_args):
    """
    A decorator on torch module creator, returns a pytorch-lightning model.

    Args:
        loss: torch loss function.
        optimizer: torch optimizer.
        **opt_args: arguments for optimizer.

    Returns: Decorator function on class or function

    """

    def from_torch(creator):

        class LightningModel(LightningModule):
            def __init__(self, *args, **kwargs):
                super().__init__()
                if not isinstance(creator, nn.Module):
                    torch_model = creator(*args, **kwargs)
                    self.copy(torch_model)

            @staticmethod
            def from_instance(instance):
                pl_model = LightningModel()
                pl_model.copy(instance)
                return pl_model

            def copy(self, torch_model):
                for k in torch_model.__dict__.keys():
                    setattr(self, k, getattr(torch_model, k))
                self.forward = torch_model.forward

            @property
            def loss(self):
                return loss

            def _forward(self, batch):
                # Handle different numbers of input for various models
                nargs = self.forward.__code__.co_argcount
                return self.forward(*(batch[:nargs - 1]))

            def training_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self._forward(batch)
                loss = self.loss(y_hat, y)
                return loss

            def validation_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self._forward(batch)
                loss = self.loss(y_hat, y)
                return loss

            def test_step(self, batch, batch_idx):
                x, y = batch
                y_hat = self._forward(batch)
                loss = self.loss(y_hat, y)
                return loss

            def configure_optimizers(self):
                return optimizer(self.parameters(), **opt_args)

        if not isinstance(creator, nn.Module):
            return LightningModel
        else:
            return LightningModel.from_instance(creator)

    return from_torch


from torch import nn

loss = nn.CrossEntropyLoss()
x = torch.ones(1, 3, 256, 256)
y = torch.zeros(1, 254, 254, dtype=torch.long)


# Case 1  decorate a module class
@to_lightning(loss, torch.optim.Adam, lr=0.01)
class Net(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3)

    # for test
    def _forward(self, x):
        return self.conv1(x)

    def forward(self, x):
        return self._forward(x)


model = Net(3, 1)
out = model.training_step([x, y], 0)
print(out.shape)

batch_size = 256
num_workers = 0
data_dir = os.path.join(os.path.dirname(__file__), "data")
from bigdl.nano.pytorch.trainer import Trainer

trainer = Trainer(max_epochs=1)
data_loader = create_data_loader(
    data_dir, batch_size, num_workers, data_transform)


# Case 2 convert from instance
def resnet18(num_classes, pretrained=True, include_top=False, freeze=True):
    backbone = vision.resnet18(pretrained=pretrained, include_top=include_top, freeze=freeze)
    output_size = backbone.get_output_size()
    head = nn.Linear(output_size, num_classes)
    return torch.nn.Sequential(backbone, head)


model: nn.Module = resnet18(10, pretrained=True, include_top=False, freeze=True)
# Save to test if load_state_dict work after conversion
torch.save(model.state_dict(), "./tmp.pth")

pl_model1 = (to_lightning(loss, torch.optim.Adam, lr=0.01)(model))
trainer.fit(pl_model1, data_loader)
trainer.test(pl_model1, data_loader)


# Case 3 decorate a function
@to_lightning(loss, torch.optim.Adam, lr=0.01)
def decorated_resnet18(num_classes, pretrained=True, include_top=False, freeze=True):
    return resnet18(num_classes, pretrained=pretrained, include_top=include_top, freeze=freeze)


pl_model2 = decorated_resnet18(10, pretrained=True, include_top=False, freeze=True)
# trainer.fit(pl_model2, data_loader)
# trainer.test(pl_model2, data_loader)

# Test if pl_model can load saved keys by torch
pl_model2.load_state_dict(torch.load('tmp.pth'), strict=True)


def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco

def preprocess(loss=None, optimizer=None, config=None, onnx=True):
    return composed(
        onnxruntime(onnx),
        to_lightning(loss, optimizer, **config)
    )


# Case 4 overall decorated
@preprocess(loss, torch.optim.Adam, {"lr": 0.01}, onnx=True)
def decorated_resnet18(num_classes, pretrained=True, include_top=False, freeze=True):
    return resnet18(num_classes, pretrained=pretrained, include_top=include_top, freeze=freeze)

pl_model3 = decorated_resnet18(10, pretrained=True, include_top=False, freeze=True)
# trainer.fit(pl_model3, data_loader)
# trainer.test(pl_model3, data_loader)

# Need to modify onnxruntime decorator
pl_model4 = preprocess(loss, torch.optim.Adam, {"lr": 0.01}, onnx=True)(model)
# trainer.fit(pl_model4, data_loader)
# trainer.test(pl_model4, data_loader)