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
from bigdl.util.common import *
from bigdl.transform.vision.image import *


class TestLayer():

    def setup_method(self, method):
        """ setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = create_spark_conf().setMaster("local[4]").setAppName("test model")
        self.sc = get_spark_context(sparkConf)
        init_engine()
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        self.image_path = os.path.join(resource_path, "pascal/000025.jpg")

    def teardown_method(self, method):
        """ teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def transformer_test(self, transformer):
        image_frame = ImageFrame.read(self.image_path)
        transformer(image_frame)
        image_frame.transform(transformer)
        image_frame.to_sample()

        image_frame = ImageFrame.read(self.image_path, self.sc)
        transformer(image_frame)
        image_frame.transform(transformer)
        sample = image_frame.to_sample()
        sample.count()

    def test_get_image(self):
        image_frame = ImageFrame.read(self.image_path)
        image_frame.get_image()

    def test_get_label(self):
        image_frame = ImageFrame.read(self.image_path)
        image_frame.get_label()

    def test_to_sample(self):
        image_frame = ImageFrame.read(self.image_path)
        image_frame.to_sample()

    def test_is_local(self):
        image_frame = ImageFrame.read(self.image_path)
        assert image_frame.is_local() is True
        image_frame = ImageFrame.read(self.image_path, self.sc)
        assert image_frame.is_local() is False

    def test_is_distributed(self):
        image_frame = ImageFrame.read(self.image_path)
        assert image_frame.is_distributed() is False
        image_frame = ImageFrame.read(self.image_path, self.sc)
        assert image_frame.is_distributed() is True

    def test_hflip(self):
        transformer = HFlip()
        self.transformer_test(transformer)

if __name__ == "__main__":
    pytest.main([__file__])
