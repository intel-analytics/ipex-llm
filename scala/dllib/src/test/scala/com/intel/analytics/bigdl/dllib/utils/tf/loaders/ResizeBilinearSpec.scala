/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.utils.tf.loaders

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.tf.Tensorflow._
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class ResizeBilinearSpec extends TensorflowSpecHelper {
  "ResizeBilinear align_corners false" should "be correct for float" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("ResizeBilinear_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("align_corners", booleanAttr(false))
        .setOp("ResizeBilinear"),
      Seq(Tensor[Float](1, 64, 64, 3).rand(), Tensor[Int](Array(224, 224), Array(2))),
      0
    )
  }

  "ResizeBilinear align_corners true" should "be correct for float" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("ResizeBilinear_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("align_corners", booleanAttr(false))
        .setOp("ResizeBilinear"),
      Seq(Tensor[Float](1, 64, 64, 3).rand(), Tensor[Int](Array(224, 224), Array(2))),
      0
    )
  }
}
