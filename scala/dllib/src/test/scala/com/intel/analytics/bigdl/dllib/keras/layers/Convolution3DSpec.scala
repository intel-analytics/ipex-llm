/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class Convolution3DSpec extends KerasBaseSpec {

  "Convolution3D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 32, 32, 32])
        |input = np.random.random([1, 3, 32, 32, 32])
        |output_tensor = Convolution3D(12, 2, 1, 3, subsample=(1, 2, 3),
        |                              dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Convolution3D[Float](12, 2, 1, 3, subsample = (1, 2, 3),
      inputShape = Shape(3, 32, 32, 32))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, precision = 1e-2)
  }

  "Convolution3D without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 16, 20, 32])
        |input = np.random.random([1, 4, 16, 20, 32])
        |output_tensor = Convolution3D(8, 2, 2, 4, activation="relu", bias=False,
        |                              border_mode="same", dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = Convolution3D[Float](8, 2, 2, 4, activation = "relu", bias = false,
      borderMode = "same", inputShape = Shape(4, 16, 20, 32))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, precision = 1e-3)
  }

}

class Convolution3DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Convolution3D[Float](12, 2, 1, 3, inputShape = Shape(3, 32, 32, 32))
    layer.build(Shape(2, 3, 32, 32, 32))
    val input = Tensor[Float](2, 3, 32, 32, 32).rand()
    runSerializationTest(layer, input)
  }
}
