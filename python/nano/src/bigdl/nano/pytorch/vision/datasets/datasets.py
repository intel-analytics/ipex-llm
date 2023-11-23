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

local_libturbo_path = None
_turbo_path = realpath(join(split(realpath(__file__))[0],
                            "../../../libs/libturbojpeg.so.0.2.0"))
if os.path.exists(_turbo_path):
    local_libturbo_path = _turbo_path
else:
    warning("libturbojpeg.so.0 not found in bigdl-nano, try to load from system.")


class ImageFolder(torchvision.datasets.ImageFolder):
    """A optimzied ImageFolder using libjpeg_turbo to load jpg images."""

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None
    ):
        """
        Create a ImageFolder.

        :param root: A string represting the root directory path.
        :param transform: A function/transform that takes in an ndarray image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        :param target_transform: A function/transform that takes in the
            target and transforms it.
        """
        super(ImageFolder, self).__init__(root, transform, target_transform)
        try:
            # it will fail if `libturbojpeg.so.0.2.0` doesn't exist
            # or load it on windows
            self.jpeg = TurboJPEG(lib_path=local_libturbo_path)
        except Exception as _e:
            warning(f"failed to load {local_libturbo_path}")
            self.jpeg = None

    def _read_image_to_bytes(self, path: str):
        with open(path, 'rb') as fd:
            img_str = fd.read()
        return img_str

    def _decode_img_libjpeg_turbo(self, img_str: str):
        bgr_array = self.jpeg.decode(img_str)
        return bgr_array

    def __getitem__(self, idx: int):
        path = self.imgs[idx][0]
        label = self.imgs[idx][1]

        if self.jpeg is not None and (path.endswith(".jpg") or path.endswith(".jpeg")):
            img_str = self._read_image_to_bytes(path)
            img = self._decode_img_libjpeg_turbo(img_str)
        else:
            img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)

        img = img.numpy()
        return img.astype('float32'), label


class SegmentationImageFolder:
    """A dataset for segmentation tasks."""

    def __init__(
            self,
            root: str,
            image_folder: str,
            mask_folder: str,
            transforms: Optional[Callable] = None,
    ):
        """
        Create dataset for image segmentation task.

        The layout the folder should look like following:
            root/image_folder/image_1.jpg
            root/image_folder/image_2.jpg
            ...
            root/mask_folder/image_1.jpg
            root/mask_folder/image_2.jpg

        image_folder and the mask_folder should have the same number images in it
        and the corresponding image and mask should have the same prefix.

        :param root: the directory to look for image_folder and mask_folder
        :param image_folder: the directory under root containing image files
        :param mask_folder: the directory under root containing mask files
        :param transforms: an optional transforms operating on images and masks
        """
        self.image_folder = os.path.join(root, image_folder)
        self.mask_folder = os.path.join(root, mask_folder)
        self.imgs = list(sorted(os.listdir(self.image_folder)))
        self.masks = list(sorted(os.listdir(self.mask_folder)))
        self.transforms = transforms
        try:
            # The following code will fail if `libturbojpeg.so.0.2.0`
            # doesn't exist or load it on windows
            self.jpeg = TurboJPEG(lib_path=local_libturbo_path)
        except Exception as _e:
            warning(f"failed to load {local_libturbo_path}")
            self.jpeg = None

    def __getitem__(self, idx: int):
        img_path = os.path.join(self.image_folder, self.imgs[idx])
        mask_path = os.path.join(self.mask_folder, self.masks[idx])

        # img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        if self.jpeg is not None and (img_path.endswith(".jpg") or img_path.endswith(".jpeg")):
            with open(img_path, 'rb') as fd:
                img = self.jpeg.decode(fd.read())
        else:
            img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.jpeg is not None and (mask_path.endswith(".jpg") or mask_path.endswith(".jpeg")):
            with open(mask_path, 'rb') as fd:
                mask = self.jpeg.decode(fd.read(), pixel_format=TJPF_GRAY)
        else:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.array(mask)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        area = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            area.append((ymax - ymin) * (xmax - xmin))

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        target["labels"] = torch.ones((num_objs,), dtype=torch.int64)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        target["image_id"] = torch.tensor([idx])
        target["area"] = torch.as_tensor(area, dtype=torch.float32)
        # suppose all instances are not crowd
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
