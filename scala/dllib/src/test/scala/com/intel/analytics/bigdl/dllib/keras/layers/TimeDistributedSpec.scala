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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class TimeDistributedSpec extends KerasBaseSpec {

  "TimeDistributed Dense" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[10, 12])
        |input = np.random.random([3, 10, 12])
        |output_tensor = TimeDistributed(Dense(8, activation="relu"))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = TimeDistributed[Float](Dense[Float](8, activation = "relu")
      .asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], inputShape = Shape(10, 12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 10, 8))
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = Array(in(0).t(), in(1))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "TimeDistributed Convolution2D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 3, 12, 12])
        |input = np.random.random([2, 4, 3, 12, 12])
        |output_tensor = TimeDistributed(Convolution2D(8, 3, 3, dim_ordering="th"))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = Sequential[Float]()
    val layer = TimeDistributed[Float](Convolution2D[Float](8, 3, 3)
      .asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]],
      inputShape = Shape(4, 3, 12, 12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4, 8, 10, 10))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, precision = 1e-3)
  }

  "TimeDistributed Convolution2D + dense" should "be the same as Keras" in {

    def converter(weights: Array[Tensor[Float]]): Array[Tensor[Float]] = {
      weights(2) = weights(2).t() // for dense
      weights
    }

    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 3, 12, 12])
        |input = np.random.random([2, 4, 3, 12, 12])
        |seq = Sequential()
        |seq.add(Convolution2D(8, 3, 3, dim_ordering="th", input_shape=[3, 12, 12]))
        |seq.add(Flatten())
        |seq.add(Dense(10))
        |output_tensor = TimeDistributed(seq)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val stepLayer = Sequential[Float]()
    stepLayer.add(Convolution2D[Float](8, 3, 3, inputShape = Shape(3, 12, 12)))
    stepLayer.add(Flatten[Float]())
    stepLayer.add(Dense[Float](10))
    val layer = TimeDistributed[Float](
      stepLayer.asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]],
      inputShape = Shape(4, 3, 12, 12))
    val seq = Sequential[Float]()
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4, 10))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter = converter, precision = 1e-3)
  }

}

class TimeDistributedSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = TimeDistributed[Float](Dense[Float](8)
      .asInstanceOf[KerasLayer[Activity, Tensor[Float], Float]], inputShape = Shape(10, 12))
    layer.build(Shape(3, 10, 12))
    val input = Tensor[Float](3, 10, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
