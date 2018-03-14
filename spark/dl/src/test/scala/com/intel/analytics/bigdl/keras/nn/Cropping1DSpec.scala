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
import com.intel.analytics.bigdl.nn.keras.{Cropping1D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class Cropping1DSpec extends KerasBaseSpec {

  "Cropping1D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[5, 6])
        |input = np.random.random([2, 5, 6])
        |output_tensor = Cropping1D((1, 2))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Cropping1D[Float]((1, 2), inputShape = Shape(5, 6))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 2, 6))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class Cropping1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Cropping1D[Float](inputShape = Shape(5, 6))
    layer.build(Shape(2, 5, 6))
    val input = Tensor[Float](2, 5, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
