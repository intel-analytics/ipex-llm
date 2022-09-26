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
from torchvision.models import resnet50
from bigdl.nano.pytorch import InferenceOptimizer
from torch.utils.data import DataLoader, Dataset

class RandomDataset(Dataset):
    """
    Random Dataset.

    This class is modified from RandomDataset in
    https://github.com/PyTorchLightning/pytorch-lightning/
    blob/master/pytorch_lightning/demos/boring_classes.py

    :param _type_ Dataset: _description_
    """

    def __init__(self, size: int, length: int):
        self.len = length
        self.data = torch.randn(length, 3, size, size)

    def __getitem__(self, index):
        return self.data[index], 0 # faked label

    def __len__(self) -> int:
        return self.len


if __name__ == "__main__":
    # 1. obtain a pre-trained ResNe50 model
    model = resnet50(pretrained=True)
    
    # 2. define a loader, here just define a random dataset
    loader = DataLoader(RandomDataset(128, 224),
                        batch_size=1)


    # 2. Accelaration inference using InferenceOptimizer
    optimizer = InferenceOptimizer()
    # optimize may take about 2 minutes to run all possible accelaration combinations
    optimizer.optimize(model=model,
                       # To obtain the latency of single sample, set batch_size=1
                       training_data=loader,
                       thread_num=1)

    # 3. Get the best model under specific restrictions or without restrictions
    acc_model, option = optimizer.get_best_model()
    print("The model with minimal latency is: ", option)

    # 4. Save (export) the accelerated model
    # The saved model files will be saved at "./best_model" directory
    save_dir = "./best_model"
    InferenceOptimizer.save(acc_model, save_dir)

    # 5. Users only need to take file in exported_files for further usage.
    if "openvino" in option:
        # You will see "ov_saved_model.bin" and "ov_saved_model.xml" under "best_model" directory
        # which contain all the required information to perform inference
        #   ov_saved_model.bin: contains the weights and biases binary data of model
        #   ov_saved_model.xml: model checkpoint for general use, describes model structure
        exported_files = [f"{save_dir}/ov_saved_model.bin", f"{save_dir}/ov_saved_model.xml"]
    elif "onnxruntime" in option:
        # You will see "onnx_saved_model.onnx" under "best_model" directory
        # which represents model checkpoint for general use, describes model structure
        exported_files = [f"{save_dir}/onnx_saved_model.onnx"]
    elif "int8" in option:
        # You will see "best_model.pt" under "best_model" directory
        # which represents model optimized by Intel® Neural Compressor
        # If the model is fully quantified, the model weights obtained are all int8
        # Otherwise, if some operations are kept at fp32, the stored weights will be mixed precision
        exported_files = [f"{save_dir}/best_model.pt"]
    elif "jit" in option:
        # You will see "ckpt.pt" under "best_model" directory
        # which stores model optimized using just-in-time compilation
        exported_files = [f"{save_dir}/ckpt.pt"]
    elif "ipex" in option or "channels_last" in option:
        # You will see "ckpt.pt" under "best_model" directory
        # saved by torch.save(model.state_dict())
        exported_files = [f"{save_dir}/ckpt.pt"]
    else:
        # typically for models of nn.Module, pl.LightningModule type
        # saved by torch.save(model.state_dict())
        exported_files = [f"{save_dir}/saved_weight.pt"]
