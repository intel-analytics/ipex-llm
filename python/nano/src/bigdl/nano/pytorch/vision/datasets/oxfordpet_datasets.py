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


from typing import Union, Callable, Optional
from turbojpeg import TurboJPEG, TJPF_GRAY
import torchvision
import cv2
import os
import numpy as np
import torch
from logging import warning
from os.path import split, join, realpath
from PIL import Image
from typing import Any

# These images in the pet dataset that don't have a proper format.
# Some of them are actually .png files instead .jpg,
# even though they are in .jpg extension.
SPECIAL_IMAGES = [
    "oxford-iiit-pet/images/Egyptian_Mau_14.jpg",
    "oxford-iiit-pet/images/Egyptian_Mau_139.jpg",
    "oxford-iiit-pet/images/Egyptian_Mau_145.jpg",
    "oxford-iiit-pet/images/Egyptian_Mau_156.jpg",
    "oxford-iiit-pet/images/Egyptian_Mau_167.jpg",
    "oxford-iiit-pet/images/Egyptian_Mau_177.jpg",
    "oxford-iiit-pet/images/Egyptian_Mau_186.jpg",
    "oxford-iiit-pet/images/Egyptian_Mau_191.jpg",
    "oxford-iiit-pet/images/Abyssinian_5.jpg",
    "oxford-iiit-pet/images/Abyssinian_34.jpg",
    "oxford-iiit-pet/images/chihuahua_121.jpg",
    "oxford-iiit-pet/images/beagle_116.jpg",
]

local_libturbo_path = None
_turbo_path = realpath(join(split(realpath(__file__))[0],
                            "../../../libs/libturbojpeg.so.0.2.0"))
if os.path.exists(_turbo_path):
    local_libturbo_path = _turbo_path
else:
    warning("libturbojpeg.so.0 not found in bigdl-nano, try to load from system.")


class OxfordIIITPet(torchvision.datasets.OxfordIIITPet):
    """A optimzied OxfordIIITPet using libjpeg_turbo to load jpg images."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ):
        """
        Create a OxfordIIITPet.

        :param root: A string represting the root directory path.
        :param transform: A function/transform that takes in an ndarray image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        :param target_transform: A function/transform that takes in the
            target and transforms it.
        :param download: If True, downloads the dataset from the internet
        """
        super(OxfordIIITPet, self).__init__(root, transform=transform,
                                            target_transform=target_transform, download=download)
        self.jpeg: Optional[TurboJPEG] = None
        self.special_images = []
        for image in SPECIAL_IMAGES:
            self.special_images.append(os.path.join(root, image))

    def _read_image_to_bytes(self, path: str):
        fd = open(path, 'rb')
        img_str = fd.read()
        fd.close()
        return img_str

    def _decode_img_libjpeg_turbo(self, img_str: str):
        if self.jpeg is None:
            self.jpeg = TurboJPEG(lib_path=local_libturbo_path)
        bgr_array = self.jpeg.decode(img_str)
        return bgr_array

    def __getitem__(self, idx: int):
        path = str(self._images[idx])
        target: Any = []
        for target_type in self._target_types:
            if target_type == "category":
                target.append(self._labels[idx])
            else:  # target_type == "segmentation"
                target.append(Image.open(self._segs[idx]))

        if not target:
            target = None
        elif len(target) == 1:
            target = target[0]
        else:
            target = tuple(target)

        if path in self.special_images:
            img = Image.open(path).convert("RGB")
        else:
            if path.endswith(".jpg") or path.endswith(".jpeg"):
                img_str = self._read_image_to_bytes(path)
                img = self._decode_img_libjpeg_turbo(img_str)
            else:
                img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img, target = self.transforms(img, target)

        img = img.numpy()
        return img.astype('float32'), target
