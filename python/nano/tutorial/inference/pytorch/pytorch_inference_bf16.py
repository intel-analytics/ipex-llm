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

    from bigdl.nano.pytorch import InferenceOptimizer

    # Inference with bfloat16
    bf16_model = InferenceOptimizer.quantize(model_ft,
                                             precision='bf16',
                                             input_sample=torch.rand(1, 3, 224, 224))
    with InferenceOptimizer.get_context(bf16_model):
        y_hat = bf16_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)

    # 'bf16' + IPEX
    ipex_model = InferenceOptimizer.quantize(model_ft,
                                             precision='bf16',
                                             use_ipex=True,
                                             input_sample=torch.rand(1, 3, 224, 224))
    with InferenceOptimizer.get_context(ipex_model):
        y_hat = ipex_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)

    # bf16 + JIT 
    jit_model = InferenceOptimizer.quantize(model_ft,
                                            precision='bf16',
                                            accelerator="jit",
                                            input_sample=torch.rand(1, 3, 224, 224))
    with InferenceOptimizer.get_context(jit_model):
        y_hat = jit_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)

    # bf16 + channels_last
    jit_model = InferenceOptimizer.quantize(model_ft,
                                            precision='bf16',
                                            channels_last=True,
                                            input_sample=torch.rand(1, 3, 224, 224))
    with InferenceOptimizer.get_context(jit_model):
        y_hat = jit_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)

    # bf16 + IPEX + JIT + channels_last
    jit_ipex_model = InferenceOptimizer.quantize(model_ft,
                                                 precision='bf16',
                                                 accelerator="jit",
                                                 use_ipex=True,
                                                 channels_last=True,
                                                 input_sample=torch.rand(1, 3, 224, 224))
    with InferenceOptimizer.get_context(jit_ipex_model):
        y_hat = jit_ipex_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)
