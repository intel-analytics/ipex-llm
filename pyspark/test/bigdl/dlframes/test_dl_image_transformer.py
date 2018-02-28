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
from bigdl.dlframes.dl_image_reader import *
from bigdl.dlframes.dl_image_transformer import *
from bigdl.util.common import *
from bigdl.transform.vision.image import *


class TestDLImageTransformer():

    def setup_method(self, method):
        """
        setup any state tied to the execution of the given method in a
        class.  setup_method is invoked for every test method of a class.
        """
        sparkConf = create_spark_conf().setMaster("local[1]").setAppName("testDLImageTransformer")
        self.sc = get_spark_context(sparkConf)
        self.sqlContext = SQLContext(self.sc)
        init_engine()
        resource_path = os.path.join(os.path.split(__file__)[0], "../resources")
        self.image_path = os.path.join(resource_path, "pascal/000025.jpg")

    def teardown_method(self, method):
        """
        teardown any state that was previously setup with a setup_method
        call.
        """
        self.sc.stop()

    def test_transform_image(self):
        image_frame = DLImageReader.readImages(self.image_path, self.sc)
        assert image_frame.count() == 1
        assert type(image_frame).__name__ == 'DataFrame'

        transformer = DLImageTransformer(
            Pipeline([Resize(256, 256), CenterCrop(224, 224),
                      ChannelNormalize(0.485, 0.456, 0.406, 0.229, 0.224, 0.225),
                      MatToTensor(), ImageFrameToSample()])
        ).setInputCol("image").setOutputCol("output")

        result = transformer.transform(image_frame)
        assert(result.count() == 1)
        first_row = result.take(1)[0][0]
        assert first_row[0].endswith("pascal/000025.jpg")
        assert first_row[1] == 375
        assert first_row[2] == 500
        assert first_row[3] == 3
        assert first_row[4] == 16
        assert len(first_row[5]) == 95959

if __name__ == "__main__":
    pytest.main([__file__])
