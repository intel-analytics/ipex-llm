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


def test_torch_Accuracy():
    from zoo.orca.learn.pytorch.pytorch_metrics import Accuracy
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
    from zoo.orca.learn.pytorch.pytorch_metrics import BinaryAccuracy
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
    from zoo.orca.learn.pytorch.pytorch_metrics import CategoricalAccuracy
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
    from zoo.orca.learn.pytorch.pytorch_metrics import SparseCategoricalAccuracy
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
    from zoo.orca.learn.pytorch.pytorch_metrics import Top5Accuracy
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
