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


# Required Dependecies

# ```bash
# pip install openvino-dev
# ```

import torch
from torchvision.models import resnet18

if __name__ == "__main__":
    model_ft = resnet18(pretrained=True)
    model_ft.eval()

    x = torch.rand(2, 3, 224, 224)

    # Accelerated Inference Using OpenVINO
    from bigdl.nano.pytorch import InferenceOptimizer
    ov_model = InferenceOptimizer.trace(model_ft,
                                        accelerator="openvino",
                                        input_sample=torch.rand(1, 3, 224, 224))

    y_hat = ov_model(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)

    # Save Optimized Model
    # The model files will be saved at "./optimized_model_ov" directory
    # There are 3 files in optimized_model_ov, users only need to take ".bin" and ".xml" file for further usage:
    #   nano_model_meta.yml: meta information of the saved model checkpoint
    #   ov_saved_model.bin: contains the weights and biases binary data of model
    #   ov_saved_model.xml: model checkpoint for general use, describes model structure
    InferenceOptimizer.save(ov_model, "./optimized_model_ov")

    # Load the Optimized Model
    loaded_model = InferenceOptimizer.load("./optimized_model_ov")

    # Inference with the Loaded Model
    with InferenceOptimizer.get_context(loaded_model):
        y_hat = loaded_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)
