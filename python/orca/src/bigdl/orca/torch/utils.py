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
from torch.utils.data.sampler import Sampler
import math
from bigdl.dllib.utils.log4Error import *


def trainable_param(model):
    training = []
    for p in model.parameters():
        if p.requires_grad:
            training.append(p)
    return training


class DistributedSequentialSampler(Sampler):
    """
    A sequential sampler used in FeatureSet when get (train=false) iterator .
    """
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset
        self.num_samples = int(math.floor(len(self.dataset) * 1.0 / num_replicas))
        extra_samples = len(self.dataset) % num_replicas
        self.epoch = 0
        if extra_samples > rank:
            self.num_samples += 1
            self.offset = self.num_samples * rank
        else:
            self.offset = self.num_samples * rank + extra_samples
        self.total_size = len(dataset)

    def __iter__(self):
        indices = list(range(self.offset, self.num_samples + self.offset))

        invalidInputError(len(indices) == self.num_samples,
                          "expect indices len match num_samples")

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
