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
        preds = torch.argmax(preds, dim=1)

    if preds.ndim == target.ndim and preds.is_floating_point():
        preds = (preds >= 0.5).long()
    return preds, target


class Accuracy:

    def __init__(self):

        self.correct = torch.tensor(0)
        self.total = torch.tensor(0)

    def __call__(self, preds, targets):
        preds, target = _unify_input_formats(preds, targets)
        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total
