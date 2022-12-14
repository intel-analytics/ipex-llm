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


from typing import Union, List

import torch
from torch.utils.data.dataloader import DataLoader

from bigdl.nano.utils.log4Error import invalidInputError, invalidOperationError


class _MultiInstanceModel(torch.nn.Module):
    def __init__(self, model, ps, send_queue, recv_queue):
        super().__init__()
        self.model = model
        self.ps = ps
        self.p_num = len(ps)
        self.send_queue = send_queue
        self.recv_queue = recv_queue

    def forward(self, input_data: Union[DataLoader, List]) -> List:
        if isinstance(input_data, (DataLoader, list)):
            length = len(input_data)
        else:
            invalidInputError(False, "The input should be a DataLoader or a list of input batchs")

        for idx_and_batch in enumerate(input_data):
            self.send_queue.put(idx_and_batch)

        outputs = [None] * length
        for _i in range(length):
            idx, output = self.recv_queue.get()
            invalidOperationError(not isinstance(output, Exception),
                                  f"forward error: {output}\ninput tensor: {input_data[idx]}")
            outputs[idx] = output

        return outputs


def _multi_instance_helper(model, recv_queue, send_queue):
    with torch.no_grad():
        while True:
            try:
                idx, args = recv_queue.get()
                if isinstance(args, tuple):
                    output = model(*args)
                else:
                    output = model(args)
                send_queue.put((idx, output))
            except Exception as e:
                send_queue.put((idx, e))
