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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.tf.Tensorflow.typeAttr
import com.intel.analytics.bigdl.utils.tf.TensorflowSpecHelper
import org.tensorflow.framework.{DataType, NodeDef}

import scala.util.Random

class Expm1Spec extends TensorflowSpecHelper {
  "Expm1" should "be correct for float" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("expm1_test")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .setOp("Expm1"),
      Seq(Tensor[Float](10).rand()),
      0
    )
  }

  "Expm1" should "be correct for double" in {
    compare[Float](
      NodeDef.newBuilder()
        .setName("expm1_test")
        .putAttr("T", typeAttr(DataType.DT_DOUBLE))
        .setOp("Expm1"),
      Seq(Tensor[Double](10).rand()),
      0
    )
  }
}

class ExpandDimsLoadTFSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val expandDim = new ExpandDimsLoadTF[Float]().setName("expandDim")
    val input = T(Tensor[Float](1, 2).apply1(_ => Random.nextFloat()),
      Tensor.scalar[Int](1))
    runSerializationTest(expandDim, input)
  }
}
