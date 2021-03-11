#
# Copyright 2018 Analytics Zoo Authors.
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
import torch


def _unify_input_formats(preds, target):
    if not (preds.ndim == target.ndim or preds.ndim == target.ndim + 1):
        raise ValueError("preds the same or one more dimensions than targets")

    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, dim=-1)

    if preds.ndim == target.ndim and preds.is_floating_point():
        preds = (preds >= 0.5).long()
    return preds, target


class Accuracy:
    """Calculates how often predictions matches labels.

    For example, if `y_true` is tensor([1, 2, 3, 4])_ and `y_pred` is tensor([0, 2, 3, 4])
    then the accuracy is 3/4 or .75.  If the weights were specified as
    tensor([1, 1, 0, 0]) then the accuracy would be 1/2 or .5.

    Usage:

    ```python
    acc = Accuracy()
    acc(torch.tensor([0, 2, 3, 4]), torch.tensor([1, 2, 3, 4]))
    assert acc.compute() == 0.75
    ```
    """

    def __init__(self):
        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)

    def __call__(self, preds, targets):
        preds, target = _unify_input_formats(preds, targets)
        self.correct += torch.sum(torch.eq(preds, targets))
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class SparseCategoricalAccuracy:
    """Calculates how often predictions matches integer labels.

    For example, if `y_true` is tensor([[2], [1]]) and `y_pred` is
    tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]) then the categorical accuracy is 1/2 or .5.
    If the weights were specified as tensor([0.7, 0.3]) then the categorical accuracy
    would be .3. You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    Usage:

    ```python
     acc = SparseCategoricalAccuracy()
     acc(torch.tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]), torch.tensor([[2], [1]]))
     assert acc.compute() == 0.5
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets):
        batch_size = targets.size(0)
        if preds.ndim == targets.ndim:
            targets = torch.squeeze(targets, dim=-1)
        preds = torch.argmax(preds, dim=-1)
        preds = preds.type_as(targets)
        self.correct += torch.sum(torch.eq(preds, targets))
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total


class CategoricalAccuracy:
    """Calculates how often predictions matches integer labels.

    For example, if `y_true` is torch.tensor([[0, 0, 1], [0, 1, 0]]) and `y_pred` is
    torch.tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]]) then the categorical accuracy is 1/2 or .5.
    If the weights were specified as tensor([0.7, 0.3]) then the categorical accuracy
    would be .3. You can provide logits of classes as `y_pred`, since argmax of
    logits and probabilities are same.

    Usage:

    ```python
    pred = torch.tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    target = torch.tensor([[0, 0, 1], [0, 1, 0]])
    cacc = CategoricalAccuracy()
    cacc(pred, target)
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets):
        batch_size = targets.size(0)
        self.correct += torch.sum(
            torch.eq(
                torch.argmax(preds, dim=-1), torch.argmax(targets, dim=-1)))
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total


class BinaryAccuracy:
    """Calculates how often predictions matches labels.

    For example, if `y_true` is tensor([1, 1, 0, 0]) and `y_pred` is tensor([0.98, 1, 0, 0.6])
    then the binary accuracy is 3/4 or .75.  If the weights were specified as
    [1, 0, 0, 1] then the binary accuracy would be 1/2 or .5.

    Usage:

    ```python
    target = torch.tensor([1, 1, 0, 0])
    pred = torch.tensor([0.98, 1, 0, 0.6])
    bac = BinaryAccuracy()
    bac(pred, target)
    assert bac.compute() == 0.75
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets, threshold=0.5):
        batch_size = targets.size(0)
        threshold = torch.tensor(threshold)
        self.correct += torch.sum(
            torch.eq(
                torch.gt(preds, threshold), targets))
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total


class Top5Accuracy:
    """Computes how often integer targets are in the top `K` predictions.

      Usage:

      ```python
      pred = torch.tensor([[0.1, 0.9, 0.8, 0.4, 0.5, 0.2],
                         [0.05, 0.95, 0, 0.4, 0.5, 0.2]])
      target = torch.tensor([2, 2])
      top5acc = Top5Accuracy()
      top5acc(pred, target)
      assert top5acc.compute() == 0.5
      ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets):
        batch_size = targets.size(0)
        _, preds = preds.topk(5, dim=-1, largest=True, sorted=True)
        preds = preds.type_as(targets).t()
        targets = targets.view(1, -1).expand_as(preds)

        self.correct += preds.eq(targets).view(-1).sum()
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total


class MAE:
    """Computes the mean absolute error between labels and predictions.

    `loss = mean(abs(y_true - y_pred), axis=-1)`

    Usage:

    ```python
    pred = torch.tensor([[1, -2], [1, 1]])
    target = torch.tensor([[0, 1], [0, 1]])
    m = MAE()
    m(pred, target)
    print(m.compute())  # tensor(1.2500)
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets, threshold=0.5):
        preds = preds.type_as(targets)
        self.correct += torch.sum(torch.abs(torch.sub(preds, targets)))
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total


class MSE:
    """Computes the mean square error between labels and predictions.

    `loss = square(abs(y_true - y_pred), axis=-1)`

    Usage:

    ```python
    pred = torch.tensor([[1, -2], [1, 1]])
    target = torch.tensor([[0, 1], [0, 1]])
    m = MSE()
    m(pred, target)
    print(m.compute())  # tensor(2.7500)
    ```
    """
    def __init__(self):
        self.total = torch.tensor(0)
        self.correct = torch.tensor(0)

    def __call__(self, preds, targets):
        preds = preds.type_as(targets)
        self.correct += torch.sum(torch.square(torch.sub(preds, targets)))
        self.total += targets.numel()

    def compute(self):
        return self.correct.float() / self.total
