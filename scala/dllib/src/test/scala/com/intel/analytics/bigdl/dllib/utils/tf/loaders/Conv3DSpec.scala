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

import java.nio.charset.Charset

import com.google.protobuf.ByteString
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.tf.{PaddingType, TensorflowSpecHelper}
import org.tensorflow.framework.{AttrValue, DataType, NodeDef}
import com.intel.analytics.bigdl.utils.tf.Tensorflow._

class Conv3DSpec extends TensorflowSpecHelper {

  "Conv3D forward with VALID padding" should "be correct" in {

    val dataFormat = AttrValue.newBuilder().setS(ByteString
      .copyFrom("NDHWC", Charset.defaultCharset())).build()

    val builder = NodeDef.newBuilder()
      .setName(s"Conv3DTest")
      .setOp("Conv3D")
      .putAttr("T", typeAttr(DataType.DT_FLOAT))
      .putAttr("strides", listIntAttr(Seq(1, 1, 2, 3, 1)))
      .putAttr("padding", PaddingType.PADDING_VALID.value)
      .putAttr("data_format", dataFormat)

    val input = Tensor[Float](4, 20, 30, 40, 3).rand()
    val filter = Tensor[Float](2, 3, 4, 3, 4).rand()

    compare[Float](
      builder,
      Seq(input, filter),
      0,
      1e-4
    )
  }

  "Conv3D forward with SAME padding" should "be correct" in {

    val dataFormat = AttrValue.newBuilder().setS(ByteString
      .copyFrom("NDHWC", Charset.defaultCharset())).build()

    val builder = NodeDef.newBuilder()
      .setName(s"Conv3DTest")
      .setOp("Conv3D")
      .putAttr("T", typeAttr(DataType.DT_FLOAT))
      .putAttr("strides", listIntAttr(Seq(1, 1, 2, 3, 1)))
      .putAttr("padding", PaddingType.PADDING_SAME.value)
      .putAttr("data_format", dataFormat)

    val input = Tensor[Float](4, 20, 30, 40, 3).rand()
    val filter = Tensor[Float](2, 3, 4, 3, 4).rand()

    compare[Float](
      builder,
      Seq(input, filter),
      0,
      1e-4
    )
  }
}
