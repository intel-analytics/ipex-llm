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
import com.intel.analytics.bigdl.nn.keras.{AveragePooling1D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class AveragePooling1DSpec extends KerasBaseSpec {

  "AveragePooling1D valid mode" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 16])
        |input = np.random.random([3, 12, 16])
        |output_tensor = AveragePooling1D()(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = AveragePooling1D[Float](inputShape = Shape(12, 16))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 6, 16))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "AveragePooling1D same mode" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[32, 32])
        |input = np.random.random([2, 32, 32])
        |output_tensor = AveragePooling1D(pool_length=3, stride=1, border_mode="same")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = AveragePooling1D[Float](poolLength = 3, stride = 1,
      borderMode = "same", inputShape = Shape(32, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 32, 32))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class AveragePooling1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = AveragePooling1D[Float](inputShape = Shape(12, 16))
    layer.build(Shape(2, 12, 16))
    val input = Tensor[Float](2, 12, 16).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
