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


from functools import partial

from torch import Tensor
from torch.nn.modules.loss import _Loss

from pytorch_lightning.lite import LightningLite


def _backward(lite: LightningLite, loss: Tensor,
              gradient=None, retain_graph=None, create_graph=False, inputs=None):
    """Used to replace Tensor's backward method."""
    # backup our backward method
    backward_backup = loss.backward
    # restore the backward method of loss to avoid infinite recursion
    setattr(loss, "backward", loss._backward)   # type: ignore
    lite.backward(loss, gradient=gradient, retain_graph=retain_graph,
                  create_graph=create_graph, inputs=inputs)
    # replace the backward method of loss with our backward method again
    setattr(loss, "backward", backward_backup)


def _forward(lite: LightningLite, loss_func: _Loss, input: Tensor, target: Tensor) -> Tensor:
    """Used to replace _Loss's forward method."""
    loss = loss_func._forward(input, target)
    # replace and save the backward method of loss
    setattr(loss, "_backward", loss.backward)
    setattr(loss, "backward", partial(_backward, lite, loss))
    return loss


def _wrap_loss_func(lite: LightningLite, loss_func: _Loss) -> _Loss:
    # replace and save the forward method of loss_func
    setattr(loss_func, "_forward", loss_func.forward)
    setattr(loss_func, "forward", partial(_forward, lite, loss_func))
    return loss_func
