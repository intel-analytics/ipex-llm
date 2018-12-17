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
import com.intel.analytics.bigdl.utils.tf.Tensorflow.{booleanAttr, floatAttr, typeAttr}
import com.intel.analytics.bigdl.utils.tf.{TensorflowDataFormat, TensorflowSpecHelper}
import org.tensorflow.framework.{DataType, NodeDef}

class FusedBatchNormSpec extends TensorflowSpecHelper {
  "FusedBatchNorm y" should "be correct when is training is true" in {
    val x = Tensor[Float](4, 8, 8, 256).rand()
    val scale = Tensor[Float](256).rand()
    val offset = Tensor[Float](256).rand()
    val mean = Tensor[Float](0)
    val variance = Tensor[Float](0)

    compare[Float](
      NodeDef.newBuilder()
        .setName("fusedbn_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("epsilon", floatAttr(0.0001f))
        .putAttr("data_format", TensorflowDataFormat.NHWC.value)
        .putAttr("is_training", booleanAttr(true))
        .setOp("FusedBatchNorm"),
      Seq(x, scale, offset, mean, variance),
      0
    )
  }

  "FusedBatchNorm batch_mean" should "be correct when is training is true" in {
    val x = Tensor[Float](4, 8, 8, 256).rand()
    val scale = Tensor[Float](256).rand()
    val offset = Tensor[Float](256).rand()
    val mean = Tensor[Float](0)
    val variance = Tensor[Float](0)

    compare[Float](
      NodeDef.newBuilder()
        .setName("fusedbn_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("epsilon", floatAttr(0.0001f))
        .putAttr("data_format", TensorflowDataFormat.NHWC.value)
        .putAttr("is_training", booleanAttr(true))
        .setOp("FusedBatchNorm"),
      Seq(x, scale, offset, mean, variance),
      1
    )
  }

  "FusedBatchNorm reserve_space_1" should "be correct when is training is true" in {
    val x = Tensor[Float](4, 8, 8, 256).rand()
    val scale = Tensor[Float](256).rand()
    val offset = Tensor[Float](256).rand()
    val mean = Tensor[Float](0)
    val variance = Tensor[Float](0)

    compare[Float](
      NodeDef.newBuilder()
        .setName("fusedbn_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("epsilon", floatAttr(0.0001f))
        .putAttr("data_format", TensorflowDataFormat.NHWC.value)
        .putAttr("is_training", booleanAttr(true))
        .setOp("FusedBatchNorm"),
      Seq(x, scale, offset, mean, variance),
      3
    )
  }

  "FusedBatchNorm batch_variance" should "be correct when is training is true" in {
    val x = Tensor[Float](4, 8, 8, 256).rand()
    val scale = Tensor[Float](256).rand()
    val offset = Tensor[Float](256).rand()
    val mean = Tensor[Float](0)
    val variance = Tensor[Float](0)

    compare[Float](
      NodeDef.newBuilder()
        .setName("fusedbn_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("epsilon", floatAttr(0.0001f))
        .putAttr("data_format", TensorflowDataFormat.NHWC.value)
        .putAttr("is_training", booleanAttr(true))
        .setOp("FusedBatchNorm"),
      Seq(x, scale, offset, mean, variance),
      2
    )
  }

  "FusedBatchNorm reserve_space_2" should "be correct when is training is true" in {
    val x = Tensor[Float](4, 8, 8, 256).rand()
    val scale = Tensor[Float](256).rand()
    val offset = Tensor[Float](256).rand()
    val mean = Tensor[Float](0)
    val variance = Tensor[Float](0)

    compare[Float](
      NodeDef.newBuilder()
        .setName("fusedbn_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("epsilon", floatAttr(0.0001f))
        .putAttr("data_format", TensorflowDataFormat.NHWC.value)
        .putAttr("is_training", booleanAttr(true))
        .setOp("FusedBatchNorm"),
      Seq(x, scale, offset, mean, variance),
      4
    )
  }

  "FusedBatchNorm" should "be correct when is training is false" in {
    val x = Tensor[Float](4, 8, 8, 256).rand()
    val scale = Tensor[Float](256).rand()
    val offset = Tensor[Float](256).rand()
    val mean = Tensor[Float](256).rand()
    val variance = Tensor[Float](256).rand().add(1f)

    compare[Float](
      NodeDef.newBuilder()
        .setName("fusedbn_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("epsilon", floatAttr(0.0001f))
        .putAttr("data_format", TensorflowDataFormat.NHWC.value)
        .putAttr("is_training", booleanAttr(false))
        .setOp("FusedBatchNorm"),
      Seq(x, scale, offset, mean, variance),
      0
    )
  }
}

