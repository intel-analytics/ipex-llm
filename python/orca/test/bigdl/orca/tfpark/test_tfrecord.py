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

import tempfile
import os.path
import pytest
from unittest import TestCase
import shutil

from bigdl.dllib.nncontext import *
from bigdl.orca.data.image import write_tfrecord, read_tfrecord


class TestTFRecord(TestCase):
    def setup_method(self, method):
        self.resource_path = os.path.join(os.path.split(__file__)[0], "../resources")

    def test_write_read_imagenet(self):
        raw_data = os.path.join(self.resource_path, "imagenet_to_tfrecord")
        temp_dir = tempfile.mkdtemp()
        try:
            write_tfrecord(format="imagenet", imagenet_path=raw_data, output_path=temp_dir)
            data_dir = os.path.join(temp_dir, "train")
            train_dataset = read_tfrecord(format="imagenet", path=data_dir, is_training=True)
            train_dataset.take(1)
        finally:
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    pytest.main([__file__])
