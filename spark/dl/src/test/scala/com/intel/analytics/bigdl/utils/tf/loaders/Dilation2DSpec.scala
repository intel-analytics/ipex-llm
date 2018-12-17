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
import com.intel.analytics.bigdl.utils.tf.{PaddingType, TensorflowDataFormat, TensorflowSpecHelper}
import org.tensorflow.framework.{DataType, NodeDef}
import com.intel.analytics.bigdl.utils.tf.Tensorflow._

class Dilation2DSpec extends TensorflowSpecHelper {
  "Dilation2D forward" should "be correct" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("Dilation2D_test")
        .setOp("Dilation2D")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("rates", listIntAttr(Seq(1, 2, 3, 1)))
        .putAttr("strides", listIntAttr(Seq(1, 3, 2, 1)))
        .putAttr("padding", PaddingType.PADDING_SAME.value),
      Seq(Tensor[Float](4, 32, 32, 3).rand(), Tensor[Float](3, 4, 3).rand()),
      0
    )

    compare[Float](
      NodeDef.newBuilder()
        .setName("Dilation2D_test")
        .setOp("Dilation2D")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("rates", listIntAttr(Seq(1, 2, 3, 1)))
        .putAttr("strides", listIntAttr(Seq(1, 3, 2, 1)))
        .putAttr("padding", PaddingType.PADDING_VALID.value),
      Seq(Tensor[Float](4, 32, 32, 3).rand(), Tensor[Float](3, 4, 3).rand()),
      0
    )
  }

  "Dilation2D forward with double model" should "be correct" in {
    compare[Double](
      NodeDef.newBuilder()
        .setName("Dilation2D_test")
        .setOp("Dilation2D")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("rates", listIntAttr(Seq(1, 2, 3, 1)))
        .putAttr("strides", listIntAttr(Seq(1, 3, 2, 1)))
        .putAttr("padding", PaddingType.PADDING_SAME.value),
      Seq(Tensor[Float](4, 32, 32, 3).rand(), Tensor[Float](3, 4, 3).rand()),
      0
    )

    compare[Double](
      NodeDef.newBuilder()
        .setName("Dilation2D_test")
        .setOp("Dilation2D")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("rates", listIntAttr(Seq(1, 2, 3, 1)))
        .putAttr("strides", listIntAttr(Seq(1, 3, 2, 1)))
        .putAttr("padding", PaddingType.PADDING_VALID.value),
      Seq(Tensor[Float](4, 32, 32, 3).rand(), Tensor[Float](3, 4, 3).rand()),
      0
    )
  }
}
