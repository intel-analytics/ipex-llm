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
import numpy as np
import os
import os.path as osp
from PIL import Image
import logging

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET


class VOCDatasets:
    """Pascal VOC dataset load

    Parameters
    ----------------
    root: str, example:'./data/VOCdevkit'.
    splits_names: tuple, ((year, trainval)).
    classes: list[str], If you using custom-voc dataset, \
    you need to config the name of custom objects.
    difficult: bool, False ignore voc xml difficult value.
    """

    def __init__(self, root="VOCdevkit",
                 splits_names=[(2007, "trainval")],
                 classes=None,
                 difficult=False) -> None:

        self.CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        if classes:
            self.CLASSES = classes

        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self._root = osp.abspath(osp.expanduser(root))
        self._diff = difficult
        self._imgid_items = self._load_items(splits_names)
        self._anno_path = osp.join('{}', 'Annotations', '{}.xml')
        self._image_path = osp.join('{}', 'JPEGImages', '{}.jpg')
        self._im_shapes = {}
        self._im_anno = [self._load_label(idx) for idx in range(len(self))]
        self._im_cache = {}

    def _load_items(self, splits_names):

        img_ids = []
        for year, txtname in splits_names:
            vocfolder = osp.join(self._root, "VOC{}".format(year))
            txtpath = osp.join(vocfolder, 'ImageSets', 'Main', txtname + '.txt')
            try:
                with open(txtpath, 'r', encoding='utf-8') as f:
                    img_ids += [(vocfolder, line.strip()) for line in f.readlines()]
            except:
                continue
        return img_ids

    def __len__(self):
        return len(self._imgid_items)

    def __iter__(self):
        img_path = [self._image_path.format(*img_id) for img_id in self._imgid_items]
        return zip(img_path, self._im_anno)

    def __getitem__(self, idx):
        img_id = self._imgid_items[idx]
        img_path = self._image_path.format(*img_id)
        if img_path in self._im_cache:
            img = self._im_cache
        else:
            img = self._read_image(img_path)

        return img, self._im_anno[idx]

    def _load_label(self, idx):
        img_id = self._imgid_items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        width = 0
        height = 0
        size = root.find('size')
        if size is not None:
            width = int(size.find('width').text)
            height = int(size.find('height').text)
        else:
            img_path = self._image_path.format(*img_id)
            img = self._read_image(img_path)
            width, height = img.size
            self._im_cache[img_path] = img

        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)

        # load label [[x1, y1, x2, y2, cls, difficult]]
        label = []
        for obj in root.iter('object'):
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.CLASSES:
                logging.warning(f"{cls_name} isn't included in {self.CLASSES}")
                continue
            cls_id = self.cat2label[cls_name]
            xml_box = obj.find('bndbox')
            xmin = float(int(xml_box.find('xmin').text) / width)
            ymin = float(int(xml_box.find('ymin').text) / height)
            xmax = float(int(xml_box.find('xmax').text) / width)
            ymax = float(int(xml_box.find('ymax').text) / height)
            label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        label = np.array(label).astype(np.float32)
        if not self._diff:
            label = label[..., :5]
        try:
            self._check_label(label, width, height)
        except AssertionError as e:
            logging.warning("Invalid label at %s, %s", anno_path, e)
        return label

    def _check_label(self, label, width=1, height=1):
        """Check if label is correct."""
        xmin = label[:, 0]
        ymin = label[:, 1]
        xmax = label[:, 2]
        ymax = label[:, 3]
        assert ((0 <= xmin) & (xmin < width)).any(), \
            "xmin must in [0, {}), given {}".format(width, xmin)
        assert ((0 <= ymin) & (ymin < height)).any(), \
            "ymin must in [0, {}), given {}".format(height, ymin)
        assert ((xmin < xmax) & (xmax <= width)).any(), \
            "xmax must in ({}, {}], given {}".format(xmin, width, xmax)
        assert ((ymin < ymax) & (ymax <= height)).any(), \
            "ymax must in ({}, {}], given {}".format(ymin, height, ymax)

    def _read_image(self, image_path):
        try:
            img = Image.open(image_path)
            img = np.array(img)
            img = img.astype(np.uint8)
            return img
        except FileNotFoundError as e:
            raise e

    @property
    def classes_label(self):
        return self.CLASSES
