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

import torch
from torch import nn
import numpy as np
from filelock import FileLock
from intel_npu_acceleration_library.backend import NNFactory
from intel_npu_acceleration_library.backend.bindings import lib as backend_lib


class LMHeadLinear(NNFactory):
    """Quantized Linear class for sliced lm_head, computing a matrix matrix multiplication
    with weights prefetching."""

    def __init__(
        self,
        inC: int,
        outC: int,
        batch: int,
        split_num: int = 2,
        profile: bool = False,
        device: str = "NPU",
        dtype: np.dtype = np.int8,
        use_split: bool = False,
    ):
        """Initialize the LMHeadLinear class.

        Args:
            inC (int): input channels
            outC (int): output channels
            batch (int): batch
            split_num (int): split in_features of lm_head to how many parts
            profile (bool): Enable/Disable profiling. Defaults to False.
            device (str): Target device, default to "NPU".
            dtype (np.dtype): weights datatype. Defaults to np.int8.

        """
        super().__init__(profile, device)
        self.inC, self.outC = inC, outC
        self.batch = batch

        self.split_num = split_num

        if use_split:
            input = self.parameter((1, self.batch, self.inC))
            res = self.dq_split_linear(input, self.split_num, self.outC, self.inC, wt_dtype=dtype,
                                       scale_factor=False)
        else:
            input = self.parameter((self.batch, self.inC))
            split_size = self.inC // split_num // 2 * 2

            for i in range(self.split_num):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < self.split_num - 1 else self.inC
                input_slice = self.slice(input, begin=[0, start_idx],
                                         end=[self.batch, end_idx])
                linear_slice = self.linear(input_slice, outC, split_size, bias=False,
                                           wt_dtype=dtype)
                if i == 0:
                    res = linear_slice
                else:
                    res += linear_slice

        print("start compiling lm_head")
        self.compile()
        print("end compiling lm_head")

    def set_weights(self, op_id, weights):
        self.set_weights_async(op_id, weights)
        with FileLock(f"lmhead_run.lock"):
            backend_lib.run(self._mm)

    def set_weights_async(self, op_id, weights):
        self.setWeights(1, op_id, *weights)

    def run(
        self, X: np.ndarray
    ) -> np.ndarray:
        """Run the layer:  $X * (W * S)^T$ .

        Args:
            X (np.ndarray): activation

        Raises:
            RuntimeError: Input, weights or scale shape mismatch

        Returns:
            np.ndarray: result
        """
        self.set_input_tensor(X, 0)
        self.elapsed = backend_lib.run(self._mm)
        if len(self.out) == 1:
            return self.out[0]
        return self.out


class SlicedLMHead(nn.Module):
    def __init__(self, weight, bias, split_num, use_split=False):
        super().__init__()
        self.split_num = split_num
        self.outC, self.inC = weight.shape
        split_size = weight.size(1) // split_num // 2 * 2
        self.lm_heads = nn.Sequential()
        for i in range(split_num):
            new_linear = torch.nn.Linear(0, 0, bias=False)
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < split_num - 1 else weight.size(1)
            new_weight = torch.nn.Parameter(weight[:, start_idx:end_idx],
                                            requires_grad=False)
            new_linear.weight = new_weight
            new_linear.in_features = new_weight.size(1)
            new_linear.out_features = new_weight.size(0)
            self.lm_heads.append(new_linear)
        self.bias = bias
        self.use_split = use_split

    def forward(self, hidden_states):
        if hidden_states.size(0) * hidden_states.size(1) == 1:
            original_shape = hidden_states.shape
            x_2d = hidden_states.view(-1, hidden_states.shape[-1])
            target_shape = tuple(list(original_shape[:-1]) + [self.outC])

            out = self.fused_lm_head.run(x_2d.numpy())
            logits = torch.from_numpy(out)
            logits = logits.view(target_shape)
        else:
            split_size = hidden_states.size(-1) // self.split_num // 2 * 2
            logits = None
            for i in range(self.split_num):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < self.split_num - 1 else self.inC
                hidden_states_slice = hidden_states[:, :, start_idx:end_idx]
                logits_slice = self.lm_heads[i](hidden_states_slice)
                if logits is None:
                    logits = logits_slice
                else:
                    logits += logits_slice

        if self.bias is None:
            return logits
        return logits + self.bias

    def get_weight_dtype(self):
        return self.lm_heads[0].weight.dtype

    def get_fused_lm_head(self):
        np_dtype = np.uint8 if self.get_weight_dtype() == torch.uint8 else np.int8
        self.fused_lm_head = LMHeadLinear(self.inC, self.outC, 1, self.split_num,
                                          False, "NPU", dtype=np_dtype, use_split=self.use_split)
        if self.use_split:
            weights = []
            scales = []
            for i in range(self.split_num):
                weights.append(self.lm_heads[i].weight)
                scales.append(self.lm_heads[i].scale)
            fused_lm_head_weights = (torch.stack(weights, axis=0).numpy(),
                                     torch.stack(scales, axis=0).numpy())
        else:
            fused_lm_head_weights = [(self.lm_heads[i].weight.data.numpy(),
                                      self.lm_heads[i].scale.data.numpy())
                                     for i in range(self.split_num)]

        self.fused_lm_head.set_weights(self.lm_heads[0].op_id,
                                       fused_lm_head_weights)
