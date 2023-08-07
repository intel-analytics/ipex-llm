//
// Copyright 2016 The BigDL Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// This would makes sure Python is aware there is more than one sub-package within bigdl,
// physically located elsewhere.
// Otherwise there would be module not found error in non-pip's setting as Python would
// only search the first bigdl package and end up finding only one sub-package.

#include <torch/extension.h>
#include <vector>

// XPU forward declarations

torch::Tensor linear_xpu_forward(
    torch::Tensor input,
    torch::Tensor weights);

// C++ interface

#define CHECK_XPU(x) TORCH_CHECK(x.device().is_xpu(), #x " must be a XPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_XPU(x); CHECK_CONTIGUOUS(x)

torch::Tensor linear_forward(
    torch::Tensor input,
    torch::Tensor weights) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);

  return linear_xpu_forward(input, weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward, "Linear forward (XPU)");
}