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
import torch
from abc import ABC, abstractmethod


def _unify_input_formats(preds, target):
    if not (preds.ndim == target.ndim or preds.ndim == target.ndim + 1):
        raise ValueError("preds the same or one more dimensions than targets")

    if preds.ndim == target.ndim + 1:
        preds = torch.argmax(preds, dim=-1)

    if preds.ndim == target.ndim and preds.is_floating_point():
        preds = (preds >= 0.5).long()
    return preds, target


def _check_same_shape(preds, targets):
    if preds.shape != targets.shape:
        raise RuntimeError("preds and targets are expected to have the same shape")


class PytorchMetric(ABC):
    """
    Base class for all pytorch metrics
    """
    @abstractmethod
    def __call__(self, preds, targets):
        pass

    @abstractmethod
    def compute(self):
        pass


class Accuracy(PytorchMetric):
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


class SparseCategoricalAccuracy(PytorchMetric):
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


class CategoricalAccuracy(PytorchMetric):
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


class BinaryAccuracy(PytorchMetric):
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


class Top5Accuracy(PytorchMetric):
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

        # torch.view requests Elements of tensors are stored
        # as a long contiguous vector in memory.
        # So need to call contiguous() before view().
        self.correct += preds.eq(targets).contiguous().view(-1).sum()
        self.total += batch_size

    def compute(self):
        return self.correct.float() / self.total


class MSE(PytorchMetric):
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
        self.sum_squared_error = torch.tensor(0.0)

    def __call__(self, preds, targets):
        _check_same_shape(preds, targets)
        self.sum_squared_error += torch.sum(torch.square(torch.sub(preds, targets)))
        self.total += targets.numel()

    def compute(self):
        return self.sum_squared_error / self.total


class MAE(PytorchMetric):
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
        self.sum_abs_error = torch.tensor(0.0)

    def __call__(self, preds, targets):
        _check_same_shape(preds, targets)
        self.sum_abs_error += torch.sum(torch.abs(torch.sub(preds, targets)))
        self.total += targets.numel()

    def compute(self):
        return self.sum_abs_error / self.total


class BinaryCrossEntropy(PytorchMetric):
    """Computes the crossentropy metric between the labels and predictions.
    This is used when there are only two labels (0 and 1).

    Usage:

    ```python
    pred = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    target = torch.tensor([[0, 1], [0, 0]])
    entropy = BinaryCrossEntropy()
    entropy(pred, target)
    assert abs(entropy.compute() - 0.81492424) < 1e-6
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.crossentropy = torch.tensor(0)

    def __call__(self, preds, targets):
        # Avoid problems with logarithm
        epsilon = 1e-7
        preds[preds <= 0] = epsilon
        preds[preds >= 1] = 1 - epsilon

        output_size = targets.view(-1).size(0)
        self.crossentropy = self.crossentropy + \
            (- targets * torch.log(preds) - (1-targets) * torch.log(1-preds)).view(-1).sum()
        self.total += output_size

    def compute(self):
        return self.crossentropy.float() / self.total


class CategoricalCrossEntropy(PytorchMetric):
    """Computes the crossentropy metric between the labels and predictions.
    This is used when there are multiple lables. The labels should be in
    the form of one-hot vectors.

    Usage:

    ```python
    pred = torch.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    target = torch.tensor([[0, 1, 0], [0, 0, 1]])
    entropy = CategoricalCrossEntropy()
    entropy(pred, target)
    assert abs(entropy.compute() - 1.1769392) < 1e-6
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.crossentropy = torch.tensor(0)

    def __call__(self, preds, targets):
        # Avoid problems with logarithm
        epsilon = 1e-7
        preds[preds <= 0] = epsilon
        preds[preds >= 1] = 1 - epsilon

        output_size = targets.size(0)
        self.crossentropy = self.crossentropy + \
            (-preds.log() * targets).sum()
        self.total += output_size

    def compute(self):
        return self.crossentropy.float() / self.total


class SparseCategoricalCrossEntropy(PytorchMetric):
    """Computes the crossentropy metric between the labels and predictions.
    This is used when there are multiple lables. The labels should be in
    the form of integers, instead of one-hot vectors.

    Usage:

    ```python
    pred = torch.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    target = torch.tensor([1, 2])
    entropy = SparseCategoricalCrossEntropy()
    entropy(pred, target)
    assert abs(entropy.compute() - 1.1769392) < 1e-6
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.crossentropy = torch.tensor(0)

    def __call__(self, preds, targets):
        # Avoid problems with logarithm
        epsilon = 1e-7
        preds[preds <= 0] = epsilon
        preds[preds >= 1] = 1 - epsilon

        output_size = targets.size(0)
        self.crossentropy = self.crossentropy + \
            (-preds.log() * torch.nn.functional.one_hot(targets)).sum()
        self.total += output_size

    def compute(self):
        return self.crossentropy.float() / self.total


class KLDivergence(PytorchMetric):
    """Computes the Kullback-Liebler divergence metric between labels and
    predictions.

    Usage:

    ```python
    pred = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    target = torch.tensor([[0, 1], [0, 0]])
    div = KLDivergence()
    div(pred, target)
    assert abs(div.compute() - 0.45814306) < 1e-6
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.divergence = torch.tensor(0)

    def __call__(self, preds, targets):
        # Avoid problems with dividing zero
        epsilon = 1e-7
        _check_same_shape(preds, targets)
        output_size = targets.size(0)
        div = targets / preds
        self.divergence = self.divergence + \
            (targets * (targets / preds + epsilon).log()).sum()
        self.total += output_size

    def compute(self):
        return self.divergence.float() / self.total


class Poisson(PytorchMetric):
    """Computes the Poisson metric between labels and
    predictions.

    Usage:

    ```python
    pred = torch.tensor([[1, 1], [0, 0]])
    target = torch.tensor([[0, 1], [0, 0]])
    poisson = Poisson()
    poisson(pred, target)
    assert abs(poisson.compute() - 0.49999997) < 1e-6
    ```
    """

    def __init__(self):
        self.total = torch.tensor(0)
        self.poisson = torch.tensor(0)

    def __call__(self, preds, targets):
        # Avoid problems with dividing zero
        epsilon = 1e-7
        _check_same_shape(preds, targets)
        output_size = targets.view(-1).size(0)
        self.poisson = self.poisson + \
            (preds - targets * torch.log(preds + epsilon)).sum()
        self.total += output_size

    def compute(self):
        return self.poisson.float() / self.total
