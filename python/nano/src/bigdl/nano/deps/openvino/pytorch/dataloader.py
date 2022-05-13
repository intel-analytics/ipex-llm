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
from openvino.tools.pot import DataLoader


class PytorchOpenVINODataLoader(DataLoader):
    def __init__(self, dataloader, collate_fn=None):
        self.dataset = dataloader.dataset
        self.collate_fn = collate_fn

    def __getitem__(self, index):
        data = self.dataset[index]
        if self.collate_fn:
            data = self.collate_fn(data)
        return data

    def __len__(self):
        return len(self.dataset)
