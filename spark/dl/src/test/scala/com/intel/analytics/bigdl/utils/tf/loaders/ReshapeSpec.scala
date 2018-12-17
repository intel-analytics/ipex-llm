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

class ReshapeSpec extends TensorflowSpecHelper {
  "Reshape" should "be correct for Float" in {
    val data = Tensor[Float](4, 32, 32, 3).rand()
    val shape = Tensor[Int](T(1, 32, 12, 32))
    compare[Float](
      NodeDef.newBuilder()
        .setName("ReshapeTest")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("Tshape", typeAttr(DataType.DT_INT32))
        .setOp("Reshape"),
      Seq(data, shape),
      0
    )
  }
  "Reshape" should "be correct for Float with inference" in {
    val data = Tensor[Float](4, 32, 32, 3).rand()
    val shape = Tensor[Int](T(1, 32, -1, 32))
    compare[Float](
      NodeDef.newBuilder()
        .setName("ReshapeTest")
        .putAttr("T", typeAttr(DataType.DT_FLOAT))
        .putAttr("Tshape", typeAttr(DataType.DT_INT32))
        .setOp("Reshape"),
      Seq(data, shape),
      0
    )
  }
}

class ReshapeLoadTFSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val reshapeLoadTF = new ReshapeLoadTF[Float]().setName("reshapeLoadTF")
    val input = T(Tensor[Float](5, 5, 5).apply1(_ => Random.nextFloat()),
      Tensor[Int](T(1, 5, 25)))
    runSerializationTest(reshapeLoadTF, input)
  }
}
