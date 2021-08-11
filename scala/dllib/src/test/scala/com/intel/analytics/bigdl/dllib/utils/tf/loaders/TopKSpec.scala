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
import com.intel.analytics.bigdl.utils.tf.Tensorflow._
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class TopKSpec extends TensorflowSpecHelper {
  "TopK" should "be correct for float tensor" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("topk_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("k", intAttr(2))
        .setOp("TopK"),
      Seq(Tensor[Float](5, 5, 6).rand()),
      0
    )

    compare[Float](
      NodeDef.newBuilder()
        .setName("topk_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("k", intAttr(2))
        .setOp("TopK"),
      Seq(Tensor[Float](5, 5, 6).rand()),
      1
    )
  }

  "TopK" should "be correct for 1D float tensor" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("topk_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("k", intAttr(2))
        .setOp("TopK"),
      Seq(Tensor[Float](5).rand()),
      0
    )

    compare[Float](
      NodeDef.newBuilder()
        .setName("topk_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("k", intAttr(2))
        .setOp("TopK"),
      Seq(Tensor[Float](5).rand()),
      1
    )
  }

  "TopK" should "be correct for float tensor when sorted is false" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("topk_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("k", intAttr(2))
        .putAttr("sorted", booleanAttr(false))
        .setOp("TopK"),
      Seq(Tensor[Float](5, 5, 6).rand()),
      0
    )

    compare[Float](
      NodeDef.newBuilder()
        .setName("topk_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("k", intAttr(2))
        .putAttr("sorted", booleanAttr(false))
        .setOp("TopK"),
      Seq(Tensor[Float](5, 5, 6).rand()),
      1
    )
  }
}

