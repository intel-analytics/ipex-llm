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
import com.intel.analytics.bigdl.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.tf.Tensorflow.{listIntAttr, typeAttr}
import com.intel.analytics.bigdl.utils.tf.{PaddingType, TensorflowDataFormat, TensorflowSpecHelper}
import org.tensorflow.framework.{DataType, NodeDef}

class DepthwiseConv2DNativeSpec extends TensorflowSpecHelper {
  "DepthwiseConv2DNative forward" should "be correct when depth multiplyer is 1" in {
    RNG.setSeed(100)
    val filter = Tensor[Float](2, 2, 3, 1).rand()
    compare[Float](
      NodeDef.newBuilder()
        .setName("depthwise_conv2d_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("padding", PaddingType.PADDING_VALID.value)
        .putAttr("data_format", TensorflowDataFormat.NHWC.value)
        .putAttr("strides", listIntAttr(Seq(1, 1, 1, 1)))
        .setOp("DepthwiseConv2dNative"),
      Seq(Tensor[Float](4, 24, 24, 3).rand(), filter),
      0
    )
  }

  "DepthwiseConv2DNative forward" should "be correct when depth multiplyer is 2" in {
    RNG.setSeed(100)
    val filter = Tensor[Float](2, 2, 3, 2).rand()
    compare[Float](
      NodeDef.newBuilder()
        .setName("depthwise_conv2d_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("padding", PaddingType.PADDING_VALID.value)
        .putAttr("data_format", TensorflowDataFormat.NHWC.value)
        .putAttr("strides", listIntAttr(Seq(1, 1, 1, 1)))
        .setOp("DepthwiseConv2dNative"),
      Seq(Tensor[Float](4, 24, 24, 3).rand(), filter),
      0
    )
  }

  "DepthwiseConv2DNative forward" should "be correct when use default data_format" in {
    RNG.setSeed(100)
    val filter = Tensor[Float](2, 2, 3, 2).rand()
    compare[Float](
      NodeDef.newBuilder()
        .setName("depthwise_conv2d_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("padding", PaddingType.PADDING_VALID.value)
        .putAttr("strides", listIntAttr(Seq(1, 1, 1, 1)))
        .setOp("DepthwiseConv2dNative"),
      Seq(Tensor[Float](4, 24, 24, 3).rand(), filter),
      0
    )
  }
}
