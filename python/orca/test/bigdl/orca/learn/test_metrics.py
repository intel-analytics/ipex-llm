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


def test_torch_Accuracy():
    from bigdl.orca.learn.pytorch.pytorch_metrics import Accuracy
    pred = torch.tensor([0, 2, 3, 4])
    target = torch.tensor([1, 2, 3, 4])
    acc = Accuracy()
    acc(pred, target)
    assert acc.compute() == 0.75
    pred = torch.tensor([0, 2, 3, 4])
    target = torch.tensor([1, 1, 2, 4])
    acc(pred, target)
    assert acc.compute() == 0.5


def test_torch_BinaryAccuracy():
    from bigdl.orca.learn.pytorch.pytorch_metrics import BinaryAccuracy
    target = torch.tensor([1, 1, 0, 0])
    pred = torch.tensor([0.98, 1, 0, 0.6])
    bac = BinaryAccuracy()
    bac(pred, target)
    assert bac.compute() == 0.75
    target = torch.tensor([1, 1, 0, 0])
    pred = torch.tensor([0.98, 1, 0, 0.6])
    bac(pred, target, threshold=0.7)
    assert bac.compute() == 0.875


def test_torch_CategoricalAccuracy():
    from bigdl.orca.learn.pytorch.pytorch_metrics import CategoricalAccuracy
    pred = torch.tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    target = torch.tensor([[0, 0, 1], [0, 1, 0]])
    cacc = CategoricalAccuracy()
    cacc(pred, target)
    assert cacc.compute() == 0.5
    pred = torch.tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    target = torch.tensor([[0, 1, 0], [0, 1, 0]])
    cacc(pred, target)
    assert cacc.compute() == 0.75


def test_torch_SparseCategoricalAccuracy():
    from bigdl.orca.learn.pytorch.pytorch_metrics import SparseCategoricalAccuracy
    pred = torch.tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    target = torch.tensor([[2], [1]])
    scacc = SparseCategoricalAccuracy()
    scacc(pred, target)
    assert scacc.compute() == 0.5
    pred = torch.tensor([[0.1, 0.9, 0.8], [0.05, 0.95, 0]])
    target = torch.tensor([2, 0])
    scacc(pred, target)
    assert scacc.compute() == 0.25


def test_torch_Top5Accuracy():
    from bigdl.orca.learn.pytorch.pytorch_metrics import Top5Accuracy
    pred = torch.tensor([[0.1, 0.9, 0.8, 0.4, 0.5, 0.2],
                         [0.05, 0.95, 0, 0.4, 0.5, 0.2]])
    target = torch.tensor([2, 2])
    top5acc = Top5Accuracy()
    top5acc(pred, target)
    assert top5acc.compute() == 0.5
    pred = torch.tensor([[0.1, 0.9, 0.8, 0.4, 0.5, 0.2],
                         [0.05, 0.95, 0, 0.4, 0.5, 0.2]])
    target = torch.tensor([[2], [1]])
    top5acc(pred, target)
    assert top5acc.compute() == 0.75


def test_torch_MAE():
    from bigdl.orca.learn.pytorch.pytorch_metrics import MAE
    pred = torch.tensor([[1, -2], [1, 1]])
    target = torch.tensor([[0, 1], [0, 1]])
    m = MAE()
    m(pred, target)
    assert m.compute() == 1.25
    pred = torch.tensor([[1, 1], [1, 1]])
    target = torch.tensor([[0, 1], [0, 1]])
    m(pred, target)
    assert m.compute() == 0.875
    pred = torch.tensor([[1.5, 2.5], [1.0, 1.0]])
    target = torch.tensor([[0.2, 1.1], [0.5, 1.0]])
    m(pred, target)
    assert abs(m.compute() - 0.85) < 1e-7   # add fault tolerance for floating point precision
    pred = torch.tensor([[1.5, 2.5, 1.5, 2.5], [1.8, 2.0, 0.5, 4.5]])
    target = torch.tensor([[0, 1, 0, 0], [0, 1, 2, 2]])
    m(pred, target)
    assert abs(m.compute() - 1.2) < 1e-7


def test_torch_MSE():
    from bigdl.orca.learn.pytorch.pytorch_metrics import MSE
    pred = torch.tensor([[1, -2], [1, 1]])
    target = torch.tensor([[1, 1], [1, 1]])
    m = MSE()
    m(pred, target)
    assert m.compute() == 2.25
    pred = torch.tensor([[1, 1], [1, 1]])
    target = torch.tensor([[1, 1], [0, 1]])
    m(pred, target)
    assert m.compute() == 1.25
    pred = torch.tensor([[1.3, 1.0], [0.2, 1.0]])
    target = torch.tensor([[1.1, 1.0], [0.0, 1.0]])
    m(pred, target)
    assert abs(m.compute() - 0.84) < 1e-7
    pred = torch.tensor([[1.2, 1.2, 1.2, 1.8], [0.2, 0.8, 0.9, 1.1]])
    target = torch.tensor([[1, 1, 1, 2], [0, 1, 1, 1]])
    m(pred, target)
    assert abs(m.compute() - 0.517) < 1e-7


def test_torch_BinaryCrossEntropy():
    from bigdl.orca.learn.pytorch.pytorch_metrics import BinaryCrossEntropy
    pred = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    target = torch.tensor([[0, 1], [0, 0]])
    entropy = BinaryCrossEntropy()
    entropy(pred, target)
    assert abs(entropy.compute() - 0.81492424) < 1e-6
    pred = torch.tensor([0.6, 0.4, 0.4, 0.6])
    target = torch.tensor([0, 1, 0, 0])
    entropy(pred, target)
    assert abs(entropy.compute() - 0.81492424) < 1e-6


def test_torch_CategoricalCrossEntropy():
    from bigdl.orca.learn.pytorch.pytorch_metrics import CategoricalCrossEntropy
    pred = torch.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    target = torch.tensor([[0, 1, 0], [0, 0, 1]])
    entropy = CategoricalCrossEntropy()
    entropy(pred, target)
    assert abs(entropy.compute() - 1.1769392) < 1e-6


def test_torch_SparseCategoricalCrossEntropy():
    from bigdl.orca.learn.pytorch.pytorch_metrics import SparseCategoricalCrossEntropy
    pred = torch.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
    target = torch.tensor([1, 2])
    entropy = SparseCategoricalCrossEntropy()
    entropy(pred, target)
    assert abs(entropy.compute() - 1.1769392) < 1e-6


def test_torch_KLDivergence():
    from bigdl.orca.learn.pytorch.pytorch_metrics import KLDivergence
    pred = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    target = torch.tensor([[0, 1], [0, 0]])
    div = KLDivergence()
    div(pred, target)
    assert abs(div.compute() - 0.45814) < 1e-5


def test_torch_Poisson():
    from bigdl.orca.learn.pytorch.pytorch_metrics import Poisson
    pred = torch.tensor([[1, 1], [0, 0]])
    target = torch.tensor([[0, 1], [0, 0]])
    poisson = Poisson()
    poisson(pred, target)
    assert abs(poisson.compute() - 0.49999997) < 1e-6
