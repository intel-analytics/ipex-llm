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

    # Normal Inference
    x = torch.rand(2, 3, 224, 224)
    model_ft.eval()
    y_hat = model_ft(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)

    # Accelerated Inference Using OpenVINO
    from bigdl.nano.pytorch import InferenceOptimizer
    ov_model = InferenceOptimizer.trace(model_ft,
                                        accelerator="openvino",
                                        input_sample=torch.rand(1, 3, 224, 224))
    
    with InferenceOptimizer.get_context(ov_model):
        y_hat = ov_model(x)
        predictions = y_hat.argmax(dim=1)
        print(predictions)

    # Save Optimized Model
    from bigdl.nano.pytorch import Trainer
    Trainer.save(ov_model, "./optimized_model")

    # Load the Optimized Model
    loaded_model = Trainer.load("./optimized_model")
    print(loaded_model(x).argmax(dim=1))
