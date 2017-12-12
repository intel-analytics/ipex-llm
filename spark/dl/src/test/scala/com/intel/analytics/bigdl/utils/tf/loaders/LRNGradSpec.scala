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

import com.intel.analytics.bigdl.nn.SpatialCrossMapLRN
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.bigdl.utils.tf.Tensorflow.{floatAttr, intAttr, typeAttr}
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

class LRNGradSpec extends TensorflowSpecHelper {
  "LRNGrad" should "be correct for float tensor" in {
    val op = SpatialCrossMapLRN[Float](7, 7, 0.5, 1, DataFormat.NHWC)
    val input = Tensor[Float](4, 8, 8, 3).rand()
    val t = op.forward(input)
    val g = Tensor[Float](4, 8, 8, 3).rand()
    compare[Float](
      NodeDef.newBuilder()
        .setName("lrn_grad_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("depth_radius", intAttr(3))
        .putAttr("beta", floatAttr(0.5f))
        .putAttr("alpha", floatAttr(1))
        .putAttr("bias", floatAttr(1))
        .setOp("LRNGrad"),
      Seq(g, input, t),
      0, 1e-2
    )
  }

  "LRNGrad" should "be correct for float tensor2" in {
    RandomGenerator.RNG.setSeed(1000)
    val op = SpatialCrossMapLRN[Float](3, 3, 1, 0, DataFormat.NHWC)
    val input = Tensor[Float](4, 8, 8, 3).rand()
    val t = op.forward(input)
    val g = Tensor[Float](4, 8, 8, 3).rand()
    compare[Float](
      NodeDef.newBuilder()
        .setName("lrn_grad_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("depth_radius", intAttr(1))
        .putAttr("beta", floatAttr(1f))
        .putAttr("alpha", floatAttr(1))
        .putAttr("bias", floatAttr(0))
        .setOp("LRNGrad"),
      Seq(g, input, t),
      0, 1e-2
    )
  }
}

