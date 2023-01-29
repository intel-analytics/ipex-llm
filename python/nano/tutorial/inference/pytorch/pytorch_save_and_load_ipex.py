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

    x = torch.rand(2, 3, 224, 224)

    # Accelerated Inference Using IPEX
    from bigdl.nano.pytorch import InferenceOptimizer
    ipex_model = InferenceOptimizer.trace(model_ft,
                                          use_ipex=True)

    # Save Optimized IPEX Model
    # The saved model files will be saved at "./optimized_model_ipex" directory
    # There are 2 files in optimized_model_ipex, users only need to take "ckpt.pth" file for further usage:
    #   nano_model_meta.yml: meta information of the saved model checkpoint
    #   ckpt.pth: pytorch state dict checkpoint for general use, describes model structure
    InferenceOptimizer.save(ipex_model, "./optimized_model_ipex")

    # Load the Optimized Model
    loaded_model = InferenceOptimizer.load("./optimized_model_ipex", model=model_ft)

    # Inference with the Loaded Model
    with InferenceOptimizer.get_context(loaded_model):
        y_hat = loaded_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)
