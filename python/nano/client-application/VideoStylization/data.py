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
# This file is adapted from https://github.com/rnwzd/FSPBT-Image-Translation/blob/master/data.py

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


from typing import Callable, Dict

import torch

from torch.utils.data import Dataset

import torchvision.transforms.functional as F
from torchvision import transforms
import pytorch_lightning as pl

from collections.abc import Iterable


# image reader writer
from pathlib import Path
from PIL import Image
from typing import Tuple


def read_image(filepath: Path, mode: str = None) -> Image:
    with open(filepath, 'rb') as file:
        image = Image.open(file)
        return image.convert(mode)


image2tensor = transforms.ToTensor()
tensor2image = transforms.ToPILImage()


def write_image(image: Image, filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(filepath))


def read_image_tensor(filepath: Path, transform=None, mode: str = 'RGB') -> torch.Tensor:
    if transform is not None:
        image = transform(read_image(filepath, mode))
    else:
        image = read_image(filepath, mode)
    return image2tensor(image)


def write_image_tensor(input: torch.Tensor, filepath: Path):
    write_image(tensor2image(input), filepath)


def get_valid_indices(H: int, W: int, patch_size: int, random_overlap: int = 0):

    vih = torch.arange(random_overlap, H-patch_size -
                       random_overlap+1, patch_size)
    viw = torch.arange(random_overlap, W-patch_size -
                       random_overlap+1, patch_size)
    if random_overlap > 0:
        rih = torch.randint_like(vih, -random_overlap, random_overlap)
        riw = torch.randint_like(viw, -random_overlap, random_overlap)
        vih += rih
        viw += riw
    vi = torch.stack(torch.meshgrid(vih, viw)).view(2, -1).t()
    return vi


def cut_patches(input: torch.Tensor, indices: Tuple[Tuple[int, int]], patch_size: int, padding: int = 0):
    # TODO use slices to get all patches at the same time ?

    patches_l = []
    for n in range(len(indices)):

        patch = F.crop(input, *(indices[n]-padding),
                       *(patch_size+padding*2,)*2)
        patches_l.append(patch)
    patches = torch.cat(patches_l, dim=0)

    return patches


def prepare_data(data_path: Path, read_func: Callable = read_image_tensor) -> Dict:
    """
    Takes a data_path of a folder which contains subfolders with input, target, etc.
    lablelled by the same names.

    :param data_path: Path of the folder containing data
    :param read_func: function that reads data and returns a tensor
    """
    data_dict = {}

    subdir_names = ["target", "input", "mask"]  # ,"helper"

    # checks only files for which there is an target
    # TODO check for images
    name_ls = [file.name for file in (
        data_path / "target").iterdir() if file.is_file()]

    subdirs = [data_path / sdn for sdn in subdir_names]
    for sd in subdirs:
        if sd.is_dir():
            data_ls = []
            files = [sd / name for name in name_ls]
            for file in files:
                tensor = read_func(file)
                H, W = tensor.shape[-2:]
                data_ls.append(tensor)
            # TODO check that all sizes match
            data_dict[sd.name] = torch.stack(data_ls, dim=0)

    data_dict['name'] = name_ls
    data_dict['len'] = len(data_dict['name'])
    data_dict['H'] = H
    data_dict['W'] = W
    return data_dict


# TODO an image is loaded whenever a patch is needed, this may be a bottleneck
class DataDictLoader():
    def __init__(self, data_dict: Dict,
                 batch_size: int = 16,
                 max_length: int = 128,
                 shuffle: bool = False):
        """

        """

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.batch_size = batch_size

        self.data_dict = data_dict
        self.dataset_len = data_dict['len']
        self.len = self.dataset_len if max_length is None else min(
            self.dataset_len, max_length)
        # Calculate # batches
        num_batches, remainder = divmod(self.len, self.batch_size)
        if remainder > 0:
            num_batches += 1
        self.num_batches = num_batches

    def __iter__(self):
        if self.shuffle:
            r = torch.randperm(self.dataset_len)
            self.data_dict = {k: v[r] if isinstance(
                v, Iterable) else v for k, v in self.data_dict.items()}
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.len:
            raise StopIteration
        batch = {k: v[self.i:self.i+self.batch_size]
                 if isinstance(v, Iterable) else v for k, v in self.data_dict.items()}

        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.num_batches


class PatchDataModule(pl.LightningDataModule):

    def __init__(self, data_dict,
                 patch_size: int = 2**5,
                 batch_size: int = 2**4,
                 patch_num: int = 2**6):
        super().__init__()
        self.data_dict = data_dict
        self.H, self.W = data_dict['H'], data_dict['W']
        self.len = data_dict['len']

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.patch_num = patch_num

    def dataloader(self, data_dict,  **kwargs):
        return DataDictLoader(data_dict, **kwargs)

    def train_dataloader(self):
        patches = self.cut_patches()
        return self.dataloader(patches, batch_size=self.batch_size, shuffle=True,
                               max_length=self.patch_num)

    def val_dataloader(self):
        return self.dataloader(self.data_dict, batch_size=1)

    def test_dataloader(self):
        return self.dataloader(self.data_dict)  # TODO batch size

    def cut_patches(self):
        # TODO cycle once
        patch_indices = get_valid_indices(
            self.H, self.W, self.patch_size, self.patch_size//4)
        dd = {k: cut_patches(
            v, patch_indices, self.patch_size) for k, v in self.data_dict.items()
            if isinstance(v, torch.Tensor)
        }
        threshold = 0.1
        mask_p = torch.mean(
            dd.get('mask', torch.ones_like(dd['input'])), dim=(-1, -2, -3))
        masked_idx = (mask_p > threshold).nonzero(as_tuple=True)[0]
        dd = {k: v[masked_idx] for k, v in dd.items()}
        dd['len'] = len(masked_idx)
        dd['H'], dd['W'] = (self.patch_size,)*2

        return dd


class ImageDataset(Dataset):
    def __init__(self, file_paths: Iterable, transform=None, read_func: Callable = read_image_tensor):
        self.file_paths = file_paths
        self.transform = transform

    def __getitem__(self, idx: int) -> dict:
        file = self.file_paths[idx]
        img = read_image_tensor(file, self.transform)
        return img, file.name

    def __len__(self) -> int:
        return len(self.file_paths)
