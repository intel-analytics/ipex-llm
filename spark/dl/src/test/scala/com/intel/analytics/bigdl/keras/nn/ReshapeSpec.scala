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
import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.keras.Reshape
import com.intel.analytics.bigdl.nn.keras.{Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class ReshapeSpec extends KerasBaseSpec {

  "Reshape" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5])
        |input = np.random.random([2, 3, 4, 5])
        |output_tensor = Reshape((4, 15))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Reshape[Float](Array(4, 15), inputShape = Shape(3, 4, 5))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4, 15))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "Reshape inference" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, ])
        |input = np.random.random([3, 12])
        |output_tensor = Reshape((-1, 2, 2))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Reshape[Float](Array(-1, 2, 2), inputShape = Shape(12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 2, 2))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class ReshapeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Reshape[Float](Array(4, 15), inputShape = Shape(3, 4, 5))
    layer.build(Shape(2, 3, 4, 5))
    val input = Tensor[Float](2, 3, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
