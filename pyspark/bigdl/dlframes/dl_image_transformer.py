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
from bigdl.util.common import JavaValue, callBigDlFunc
from pyspark.ml.param.shared import *
from pyspark.ml.wrapper import JavaTransformer

if sys.version >= '3':
    long = int
    unicode = str

class DLImageTransformer(JavaTransformer, HasInputCol, HasOutputCol, JavaValue):
    """
    Provides DataFrame-based API for image pre-processing and feature transformation.
    DLImageTransformer follows the Spark Transformer API pattern and can be used as one stage
    in Spark ML pipeline.

    The input column can be either DLImageSchema.byteSchema or DLImageSchema.floatSchema. If
    using DLImageReader, the default format is DLImageSchema.byteSchema
    The output column is always DLImageSchema.floatSchema.
    """

    def __init__(self,  transformer, jvalue=None, bigdl_type="float"):
        super(DLImageTransformer, self).__init__()
        self.value = jvalue if jvalue else callBigDlFunc(
            bigdl_type, self.jvm_class_constructor(), transformer)
        self._java_obj = self.value
        self.bigdl_type = bigdl_type

    def transform(self, dataset):
        """
        Apply the transformer to the images in "inputCol" and store the transformed result
        into "outputCols"
        """
        self._transfer_params_to_java()
        return callBigDlFunc(self.bigdl_type, "dlImageTransform", self.value, dataset)


