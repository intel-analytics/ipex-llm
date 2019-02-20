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

import scala.util.Random

class ConvLSTM2DSpec extends KerasBaseSpec {

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    Array(in(6), in(8), in(7),
      in(0), in(2), in(1),
      in(3), in(5), in(4),
      in(9), in(11), in(10))
  }

  "ConvLSTM2D return sequences with same padding" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[8, 40, 40, 32])
        |input = np.random.random([4, 8, 40, 40, 32])
        |output_tensor = ConvLSTM2D(32, 4, 4, return_sequences=True,
        |                           dim_ordering="th", border_mode="same")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = ConvLSTM2D[Float](32, 4, returnSequences = true, borderMode = "same",
      inputShape = Shape(8, 40, 40, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 32, 40, 32))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, precision = 1e-2)
  }

  "ConvLSTM2D return sequences with valid padding" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[8, 40, 40, 32])
        |input = np.random.random([4, 8, 40, 40, 32])
        |output_tensor = ConvLSTM2D(32, 4, 4, return_sequences=True,
        |                           dim_ordering="th", border_mode="valid")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = ConvLSTM2D[Float](32, 4, returnSequences = true, borderMode = "valid",
      inputShape = Shape(8, 40, 40, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 32, 37, 29))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, precision = 1e-2)
  }

  "ConvLSTM2D go backwards with same padding" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 8, 16, 16])
        |input = np.random.random([4, 4, 8, 16, 16])
        |output_tensor = ConvLSTM2D(8, 2, 2, go_backwards=True,
        |                           inner_activation="sigmoid",
        |                           dim_ordering="th", border_mode="same")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = ConvLSTM2D[Float](8, 2, goBackwards = true, borderMode = "same",
      innerActivation = "sigmoid", inputShape = Shape(4, 8, 16, 16))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 16, 16))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, precision = 1e-2)
  }

  "ConvLSTM2D go backwards with valid padding" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 8, 16, 16])
        |input = np.random.random([4, 4, 8, 16, 16])
        |output_tensor = ConvLSTM2D(8, 2, 2, go_backwards=True,
        |                           inner_activation="sigmoid",
        |                           dim_ordering="th", border_mode="valid")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = ConvLSTM2D[Float](8, 2, goBackwards = true, borderMode = "valid",
      innerActivation = "sigmoid", inputShape = Shape(4, 8, 15, 15))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 14, 14))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, precision = 1e-2)
  }
}

class ConvLSTM2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ConvLSTM2D[Float](32, 4, inputShape = Shape(8, 40, 40, 32))
    layer.build(Shape(2, 8, 40, 40, 32))
    val input = Tensor[Float](2, 8, 40, 40, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
