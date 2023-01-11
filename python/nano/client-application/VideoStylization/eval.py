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
#
# This file is adapted from https://github.com/rnwzd/FSPBT-Image-Translation/blob/master/eval.py

# MIT License

# Copyright (c) 2022 Lorenzo Breschi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from data import write_image_tensor, ImageDataset

from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from train import data_path, model_save_path

# you can overwrite data_path here
output_dir = data_path/'output'
input_dir = data_path/'input'

# Change these depending on your hardware, has to match training settings
device = 'cpu'
dtype = torch.float32

generator = torch.load(model_save_path/"generator.pt")
generator.eval()
generator.to(device, dtype)


# TODO batch size, async dataloader
file_paths = [file for file in input_dir.iterdir()]


params = {'batch_size': 1,
          'num_workers': 8,
          'pin_memory': True}

dataset = ImageDataset(file_paths, transform=None)
loader = DataLoader(dataset, **params)

# TODO multiprocess and asynchronous writing of files

import time
st = time.perf_counter()
with torch.no_grad():
    for inputs, names in tqdm(loader):
        inputs = inputs.to(device, dtype)
        outputs = generator(inputs)
        del inputs
        for j in range(len(outputs)):
            write_image_tensor(outputs[j], output_dir/names[j])
        del outputs
end = time.perf_counter()
print(f"eval cost {end-st}s.")
