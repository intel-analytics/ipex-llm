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
    loss.backward = loss._backward      # type: ignore
    lite.backward(loss, gradient=gradient, retain_graph=retain_graph,
                  create_graph=create_graph, inputs=inputs)
    # replace the backward method of loss with our backward method again
    loss.backward = backward_backup     # type: ignore


class _LossFuncWrapper(_Loss):
    def __init__(self, lite: LightningLite, loss_func: _Loss):
        super().__init__()
        self._lite = lite
        self._loss_func = loss_func

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = self._loss_func(input, target)
        # replace and save the backward method of loss
        loss._backward = loss.backward
        loss.backward = partial(_backward, self._lite, loss)
        return loss
