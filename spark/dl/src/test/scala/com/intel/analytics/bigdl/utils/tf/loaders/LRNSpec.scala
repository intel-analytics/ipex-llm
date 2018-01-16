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
import com.intel.analytics.bigdl.utils.tf.Tensorflow.{floatAttr, intAttr, typeAttr}
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class LRNSpec extends TensorflowSpecHelper {
  "LRN" should "be correct for float tensor" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("lrn_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("depth_radius", intAttr(3))
        .putAttr("beta", floatAttr(0.5f))
        .putAttr("alpha", floatAttr(1))
        .putAttr("bias", floatAttr(1))
        .setOp("LRN"),
      Seq(Tensor[Float](4, 8, 8, 3).rand()),
      0
    )
  }

  "LRN" should "be correct for float tensor2" in {
    val t = Tensor[Float](4, 8, 8, 3).fill(1f)
    compare[Float](
      NodeDef.newBuilder()
        .setName("lrn_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("depth_radius", intAttr(1))
        .putAttr("beta", floatAttr(1f))
        .putAttr("alpha", floatAttr(1))
        .putAttr("bias", floatAttr(0))
        .setOp("LRN"),
      Seq(t),
      0
    )
  }
}

