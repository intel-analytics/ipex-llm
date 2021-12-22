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


import pytest
import os
from unittest import TestCase
from bigdl.nano.pytorch.vision.models import vision
from test.pytorch.utils._train_imagefolder import train_torch_lightning


batch_size = 1
resources_root = os.path.join(os.path.dirname(__file__), "resources")
root_dir1 = os.path.join(resources_root, "train_image_folder_png")
root_dir2 = os.path.join(resources_root, "train_image_folder_jpg")


class TestImageFolder(TestCase):

    def test_resnet18_quantitrain_image_folder_pngze(self):
        resnet18 = vision.resnet18(
            pretrained=False, include_top=False, freeze=True)
        train_torch_lightning(resnet18, root_dir1, batch_size)
        train_torch_lightning(resnet18, root_dir2, batch_size)


if __name__ == '__main__':
    pytest.main([__file__])
