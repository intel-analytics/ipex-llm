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
from typing import Sequence
from unittest import TestCase

import torch
from bigdl.nano.utils.pytorch.data_adapter import DataAdapter
from bigdl.nano.utils.pytorch.model_info import ModelInfo
from torch import nn


def compare_tensor(a, b):
    try:
        if isinstance(a, Sequence):
            if len(a) == len(b):
                tmp_result = True
                for i in range(len(a)):
                    tmp_result = tmp_result and compare_tensor(a[i], b[i])
                return tmp_result
            else:
                return False
        elif isinstance(a, torch.Tensor):
            return a.equal(b)
        else:
            return a == b
    except Exception as e:
        return False


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 10)
        self.linear3 = nn.Linear(10, 10)

    def forward(self, input1: torch.Tensor = None, input2: torch.Tensor = None, input3=None):
        a = self.linear1(input1)
        b = self.linear2(input2)
        c = self.linear3(input3)
        d = torch.add(torch.add(a, b), c)
        d = torch.flatten(torch.sum(d))
        return d


class TestDispatcherPytorch(TestCase):
    def setUp(self) -> None:
        self.model = TestNet()
        self.model_info = ModelInfo(self.model)
        self.data_adapter = DataAdapter(self.model)

    def test_model_info(self):
        self.assertEqual(tuple(self.model_info.forward_args), tuple(['input1', 'input2', 'input3']))
        self.assertEqual(tuple(self.model_info.forward_defaults), tuple([None, None, None]))
        self.assertEqual(self.model_info.forward_annotations,
                         {'input1': torch.Tensor,
                          'input2': torch.Tensor})

    def test_get_data(self):
        tmp = torch.ones(10)
        tmp_label = torch.ones(1)
        self.assertEqual(tuple(self.data_adapter.get_data(tmp)), tuple([tmp, tmp_label]))
        self.assertEqual(tuple(self.data_adapter.get_data([tmp])), tuple([tmp, tmp_label]))
        self.assertEqual(tuple(self.data_adapter.get_data([tmp, tmp])),
                         tuple([tmp, tmp]))
        self.assertEqual(tuple(self.data_adapter.get_data([[tmp, tmp, tmp], [tmp_label]])),
                         tuple([[tmp, tmp, tmp], [tmp_label]]))
        self.assertEqual(tuple(self.data_adapter.get_data([[tmp, tmp, tmp], []])),
                         tuple([[tmp, tmp, tmp], []]))
        self.assertEqual(tuple(self.data_adapter.get_data([tmp, tmp, tmp])),
                         tuple([(tmp, tmp, tmp), tuple()]))
        self.assertEqual(tuple(self.data_adapter.get_data([tmp, tmp, tmp, tmp_label])),
                         tuple([(tmp, tmp, tmp), tuple([tmp_label])]))

    def test_get_dataloader(self):
        tmp = torch.ones(10)
        tmp_label = torch.ones(1)
        result = next(iter(self.data_adapter.get_dataloader(tmp)))
        self.assertTrue(compare_tensor(result, (tmp, tmp_label)))
        result = next(iter(self.data_adapter.get_dataloader([tmp])))
        self.assertTrue(compare_tensor(result, (tmp, tmp_label)))
        result = next(iter(self.data_adapter.get_dataloader([tmp, tmp])))
        self.assertTrue(compare_tensor(result, tuple([tmp, tmp])))
        result = next(iter(self.data_adapter.get_dataloader([[tmp, tmp, tmp], [tmp_label]])))
        self.assertTrue(compare_tensor(result, tuple([[tmp, tmp, tmp], [tmp_label]])))
        result = next(iter(self.data_adapter.get_dataloader([[tmp, tmp, tmp], []])))
        self.assertTrue(compare_tensor(result, tuple([[tmp, tmp, tmp], []])))
        result = next(iter(self.data_adapter.get_dataloader([tmp, tmp, tmp])))
        self.assertTrue(compare_tensor(result, tuple([(tmp, tmp, tmp), tuple()])))
        result = next(iter(self.data_adapter.get_dataloader([tmp, tmp, tmp, tmp_label])))
        self.assertTrue(compare_tensor(result, tuple([(tmp, tmp, tmp), tuple([tmp_label])])))

    def test_complement_input_sample(self):
        tmp = torch.ones(10)
        self.assertEqual(tuple(self.data_adapter.complement_input_sample([])),
                         tuple((None, None, None)))
        self.assertEqual(tuple(self.data_adapter.complement_input_sample([tmp])),
                         tuple((tmp, None, None)))
        self.assertEqual(tuple(self.data_adapter.complement_input_sample([tmp, tmp])),
                         tuple((tmp, tmp, None)))
        self.assertEqual(tuple(self.data_adapter.complement_input_sample([tmp, tmp, tmp])),
                         tuple((tmp, tmp, tmp)))
