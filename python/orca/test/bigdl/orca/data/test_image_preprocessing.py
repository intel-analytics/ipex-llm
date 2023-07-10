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

import os.path
import pytest
from unittest import TestCase
import bigdl.orca.data
from bigdl.orca.data.image.preprocessing import read_images_pil, read_images_spark

from bigdl.dllib.nncontext import *
import PIL


class TestImagePreprocessing(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources")

    def test_read_images(self):
        file_path = os.path.join(self.resource_path, "cats/")
        data_shard = bigdl.orca.data.read_images(file_path)
        collected = data_shard.collect()
        size = (80, 80)
        resized = data_shard.rdd.map(lambda x: x.resize(size))
        for im in collected:
            self.assertTrue(isinstance(im, PIL.Image.Image))
        self.assertTrue(data_shard.rdd.count() == 6)
        self.assertTrue(resized.count() == 6)

    def test_read_images_pil(self):
        file_path = os.path.join(self.resource_path, "cats/")
        data_shard = bigdl.orca.data.read_images(file_path, backend="pillow")
        collected = data_shard.collect()
        size = (80, 80)
        resized = data_shard.rdd.map(lambda x: x.resize(size))
        for im in collected:
            self.assertTrue(isinstance(im, PIL.Image.Image))
        self.assertTrue(data_shard.rdd.count() == 6)
        self.assertTrue(resized.count() == 6)

    def test_read_images_pil_withlabel(self):
        file_path = os.path.join(self.resource_path, "cats/")

        def get_label(file_name):
            label = 1 if 'dog' in file_name.split('/')[-1] else 0
            return label

        data_shard = bigdl.orca.data.read_images(file_path, get_label, backend="pillow")
        collected = data_shard.collect()
        size = (80, 80)
        resized = data_shard.rdd.map(lambda x: (x[0].resize(size), x[1]))
        for im in collected:
            self.assertTrue(isinstance(im, tuple) and
                            isinstance(im[0], PIL.Image.Image) and
                            im[1] == 0)
        self.assertTrue(data_shard.rdd.count() == 6)
        self.assertTrue(resized.count() == 6)

    def test_read_images_spark(self):
        file_path = os.path.join(self.resource_path, "cats/")
        data_shard = bigdl.orca.data.read_images(file_path, backend="spark")
        collected = data_shard.collect()
        size = (80, 80)
        resized = data_shard.rdd.map(lambda x: x.resize(size))
        for im in collected:
            self.assertTrue(isinstance(im, PIL.Image.Image))
        self.assertTrue(data_shard.rdd.count() == 6)
        self.assertTrue(resized.count() == 6)

    def test_read_images_spark_withlabel(self):
        file_path = os.path.join(self.resource_path, "dogs/")

        def get_label(file_name):
            label = 1 if 'dog' in file_name.split('/')[-1] else 0
            return label

        data_shard = bigdl.orca.data.read_images(file_path, get_label, backend="spark")
        collected = data_shard.collect()
        for im in collected:
            self.assertTrue(isinstance(im, tuple) and
                            isinstance(im[0], PIL.Image.Image) and
                            im[1] == 1)
        self.assertTrue(data_shard.rdd.count() == 6)

    def test_read_images_pil_with_masks(self):
        image_path = os.path.join(self.resource_path, "tsg_salt/images")
        target_path = os.path.join(self.resource_path, "tsg_salt/masks")

        data_shard = bigdl.orca.data.read_images(image_path,
                                                 target_path=target_path,
                                                 image_type='.png',
                                                 target_type='.png')
        print(len(data_shard))
        collected = data_shard.collect()
        for im in collected:
            self.assertTrue(isinstance(im, tuple) and
                            isinstance(im[0], PIL.Image.Image) and
                            isinstance(im[1], PIL.Image.Image))
        self.assertTrue(data_shard.rdd.count() == 5)

    def test_read_images_spark_with_masks(self):
        image_path = os.path.join(self.resource_path, "tsg_salt/images")
        target_path = os.path.join(self.resource_path, "tsg_salt/masks")

        data_shard = bigdl.orca.data.read_images(image_path,
                                                 target_path=target_path,
                                                 image_type='.png',
                                                 target_type='.png',
                                                 backend="spark")
        print(len(data_shard))
        collected = data_shard.collect()
        for im in collected:
            self.assertTrue(isinstance(im, tuple) and
                            isinstance(im[0], PIL.Image.Image) and
                            isinstance(im[1], PIL.Image.Image))
        self.assertTrue(data_shard.rdd.count() == 5)

    def test_read_voc(self):
        from bigdl.orca.data.image.preprocessing import read_voc
        image_path = os.path.join(self.resource_path, "VOCdevkit")
        data_shard = read_voc(image_path,
                              split_names=[(2007, "trainval")],
                              max_samples=5)

        collected = data_shard.collect()
        print(collected)
        for im in collected:
            self.assertTrue(isinstance(im, tuple) and
                            isinstance(im[0], PIL.Image.Image) and
                            isinstance(im[1], np.ndarray))
        self.assertTrue(data_shard.rdd.count() == 5)


if __name__ == "__main__":
    pytest.main([__file__])
