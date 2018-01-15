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
import org.tensorflow.framework.{AttrValue, DataType, NodeDef}
import com.intel.analytics.bigdl.utils.tf.Tensorflow._

abstract class BinaryOpBaseSpec extends TensorflowSpecHelper {

  def getOpName: String

  def getInputs: Seq[Tensor[_]]

  def getAttrs: Seq[(String, AttrValue)] = Seq.empty

  def compareExactly: Boolean = false

  s"$getOpName forward with float model" should "be correct" in {

    val builder = NodeDef.newBuilder()
      .setName(s"${getOpName}Test")
      .setOp(getOpName)
      .putAttr("T", typeAttr(DataType.DT_FLOAT))

    for ((k, v) <- getAttrs) {
      builder.putAttr(k, v)
    }

    if (!compareExactly) {
      compare[Float](
        builder,
        getInputs,
        0
      )
    } else {
      val (bigdl, tf) = getResult[Float, Float](builder, getInputs, 0)
      bigdl should be (tf)
    }
  }

  s"$getOpName forward with double model" should "be correct" in {

    val builder = NodeDef.newBuilder()
      .setName(s"${getOpName}Test")
      .setOp(getOpName)
      .putAttr("T", typeAttr(DataType.DT_FLOAT))

    for ((k, v) <- getAttrs) {
      builder.putAttr(k, v)
    }

    if (!compareExactly) {
      compare[Double](
        builder,
        getInputs,
        0
      )
    } else {
      val (bigdl, tf) = getResult[Double, Float](builder, getInputs, 0)
      bigdl should be (tf)
    }
  }
}
