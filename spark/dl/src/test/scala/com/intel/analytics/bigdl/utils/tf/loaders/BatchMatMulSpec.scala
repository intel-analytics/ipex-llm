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

class BatchMatMulSpec extends TensorflowSpecHelper {
  import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

  "BatchMatMul with two dim forward" should "be correct" in {
    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(false))
        .putAttr("adj_y", booleanAttr(false)),
      Seq(Tensor[Float](4, 3).rand(), Tensor[Float](3, 4).rand()),
      0
    )

    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(true))
        .putAttr("adj_y", booleanAttr(true)),
      Seq(Tensor[Float](4, 3).rand(), Tensor[Float](3, 4).rand()),
      0
    )

    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(false))
        .putAttr("adj_y", booleanAttr(true)),
      Seq(Tensor[Float](4, 3).rand(), Tensor[Float](4, 3).rand()),
      0
    )
  }

  "BatchMatMul with three dim forward" should "be correct" in {
    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(false))
        .putAttr("adj_y", booleanAttr(false)),
      Seq(Tensor[Float](4, 3, 2).rand(), Tensor[Float](4, 2, 3).rand()),
      0
    )

    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(true))
        .putAttr("adj_y", booleanAttr(true)),
      Seq(Tensor[Float](4, 3, 2).rand(), Tensor[Float](4, 2, 3).rand()),
      0
    )

    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(false))
        .putAttr("adj_y", booleanAttr(true)),
      Seq(Tensor[Float](4, 3, 2).rand(), Tensor[Float](4, 3, 2).rand()),
      0
    )
  }

  "BatchMatMul with more dim forward" should "be correct" in {
    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(false))
        .putAttr("adj_y", booleanAttr(false)),
      Seq(Tensor[Float](2, 4, 3, 2).rand(), Tensor[Float](2, 4, 2, 3).rand()),
      0
    )

    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(true))
        .putAttr("adj_y", booleanAttr(true)),
      Seq(Tensor[Float](2, 4, 3, 2).rand(), Tensor[Float](2, 4, 2, 3).rand()),
      0
    )

    compare(
      NodeDef.newBuilder()
        .setName("BatchMatMul_test")
        .setOp("BatchMatMul")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("adj_x", booleanAttr(false))
        .putAttr("adj_y", booleanAttr(true)),
      Seq(Tensor[Float](2, 4, 3, 2).rand(), Tensor[Float](2, 4, 3, 2).rand()),
      0
    )
  }
}
