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
# pip install onnx onnxruntime
# ```

import torch

if __name__ == "__main__":

    # >>> import a pretrained ResNet-18 model >>>
    from torchvision.models import resnet18

    model_ft = resnet18(pretrained=True)
    # <<< import a pretrained ResNet-18 model <<<

    # Normal Inference
    x = torch.rand(2, 3, 224, 224)
    # >>> set the model in evaluation mode >>>
    model_ft.eval()
    # <<< set the model in evaluation mode <<<
    y_hat = model_ft(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)

    # Accelerated Inference Using ONNX Runtime
    # >>> import BigDL-Nano Trainer, and trace the model for acceleration >>>
    from bigdl.nano.pytorch import Trainer

    ort_model = Trainer.trace(model_ft,
                              accelerator="onnxruntime",
                              input_sample=torch.rand(1, 3, 224, 224))
    # <<< import BigDL-Nano Trainer, and trace the model for acceleration <<<

    # >>> do normal inference steps with the optimized model >>>
    x = torch.rand(2, 3, 224, 224)
    # use the optimized model here
    y_hat = ort_model(x)
    predictions = y_hat.argmax(dim=1)
    print(predictions)
    # <<< do normal inference steps with the optimized model <<<

    # Save Optimized Model
    Trainer.save(ort_model, "./optimized_model")

    # Load the Optimized Model
    loaded_model = Trainer.load("./optimized_model")
    print(loaded_model(x).argmax(dim=1))
