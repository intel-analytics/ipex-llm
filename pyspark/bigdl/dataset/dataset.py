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

import sys
from bigdl.util.common import JavaValue
from bigdl.util.common import callBigDlFunc
from bigdl.util.common import *
from bigdl.transform.vision.image import *

if sys.version >= '3':
    long = int
    unicode = str

class DataSet(JavaValue):

    def __init__(self, jvalue=None, image_frame = None, bigdl_type="float"):
        self.bigdl_type = bigdl_type
        if jvalue:
            self.value = jvalue
        if image_frame:
            self.image_frame = image_frame

    @classmethod
    def image_frame(cls, image_frame, bigdl_type="float"):
        jvalue = callBigDlFunc(bigdl_type, "createDatasetFromImageFrame", image_frame)
        return DataSet(jvalue=jvalue, image_frame = image_frame)

    def transform(self, transformer):
        if isinstance(transformer, FeatureTransformer):
            jvalue = callBigDlFunc(self.bigdl_type, "featureTransformDataset", self.value, transformer)
            return DataSet(jvalue=jvalue)
    def get_image_frame(self):
        return self.image_frame