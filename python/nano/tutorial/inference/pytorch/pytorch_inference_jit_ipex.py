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
from torchvision.models import resnet18

if __name__ == "__main__":

    model_ft = resnet18(pretrained=True)

    # Normal Inference
    x = torch.rand(2, 3, 224, 224)
    y_hat = model_ft(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)

    # Accelerated Inference Using JIT/IPEX/JIT+IPEX
    # it is recommended to use JIT and IPEX together
    from bigdl.nano.pytorch import InferenceOptimizer

    # JIT
    jit_model = InferenceOptimizer.trace(model_ft,
                                         accelerator="jit",
                                         input_sample=torch.rand(1, 3, 224, 224))
    with InferenceOptimizer.get_context(jit_model):
        y_hat = jit_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)

    # IPEX
    ipex_model = InferenceOptimizer.trace(model_ft,
                                          use_ipex=True)
    with InferenceOptimizer.get_context(ipex_model):
        y_hat = ipex_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)

    # IPEX + JIT
    jit_ipex_model = InferenceOptimizer.trace(model_ft,
                                              accelerator="jit",
                                              use_ipex=True,
                                              input_sample=torch.rand(1, 3, 224, 224))
    with InferenceOptimizer.get_context(jit_ipex_model):
        y_hat = jit_ipex_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)
