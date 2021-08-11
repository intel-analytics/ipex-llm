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
import com.intel.analytics.bigdl.nn.keras.Masking
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class MaskingSpec extends KerasBaseSpec{

  "Masking" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3])
        |input = np.random.uniform(0, 1, [1, 3])
        |output_tensor = Masking(0.0)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val masking = Masking[Float](0.0, inputShape = Shape(3))
    seq.add(masking)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "Masking 3D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24])
        |input = np.random.random([2, 3, 24])
        |output_tensor = Masking(0.0)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val masking = Masking[Float](0.0, inputShape = Shape(3, 24))
    seq.add(masking)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class MaskingSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Masking[Float](0.0, inputShape = Shape(3, 12))
    layer.build(Shape(2, 3, 12))
    val input = Tensor[Float](2, 3, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
