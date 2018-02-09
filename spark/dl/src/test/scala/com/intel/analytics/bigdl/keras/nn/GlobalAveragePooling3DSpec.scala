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

package com.intel.analytics.bigdl.keras.nn

import com.intel.analytics.bigdl.keras.KerasBaseSpec
import com.intel.analytics.bigdl.nn.keras.{Sequential => KSequential}
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.GlobalAveragePooling3D
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class GlobalAveragePooling3DSpec extends KerasBaseSpec{

  "GlobalAveragePooling3D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5, 6])
        |input = np.random.random([2, 3, 4, 5, 6])
        |output_tensor = GlobalAveragePooling3D(dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = GlobalAveragePooling3D[Float](inputShape = Shape(3, 4, 5, 6))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class GlobalAveragePooling3DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = GlobalAveragePooling3D[Float](inputShape = Shape(3, 4, 5, 6))
    layer.build(Shape(2, 3, 4, 5, 6))
    val input = Tensor[Float](2, 3, 4, 5, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
