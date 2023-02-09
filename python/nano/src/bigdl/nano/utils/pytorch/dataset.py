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
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate


class RepeatDataset(Dataset):
    def __init__(self, sample, num=1):
        self.data = sample
        self.length = num

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data


def remove_batch_dim_fn(loader):
    def warpper_fn(batch):
        data = default_collate(batch)

        def recusive_remove(data):
            if isinstance(data, torch.Tensor):
                return data.squeeze(0)
            else:
                return tuple([recusive_remove(x) for x in data])
        return recusive_remove(data)
    loader.collate_fn = warpper_fn
    return loader
