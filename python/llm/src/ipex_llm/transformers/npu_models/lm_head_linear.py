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
        group_size: int = 0,
        asym: bool = False,
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
                                       scale_factor=(group_size == 0), asym=asym)
        else:
            input = self.parameter((self.batch, self.inC))
            split_size = self.inC // split_num // 2 * 2

            for i in range(self.split_num):
                start_idx = i * split_size
                end_idx = (i + 1) * split_size if i < self.split_num - 1 else self.inC
                input_slice = self.slice(input, begin=[0, start_idx],
                                         end=[self.batch, end_idx])
                linear_slice = self.linear(input_slice, outC, split_size, bias=False,
                                           wt_dtype=dtype, asym=asym)
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
