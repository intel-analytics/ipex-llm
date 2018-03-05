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
import com.intel.analytics.bigdl.nn.keras.{Dense, SimpleRNN, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class SimpleRNNSpec extends KerasBaseSpec {

  "SimpleRNN computeOutputShape" should "work properly" in {
    val seq = KSequential[Float]()
    val rnn = SimpleRNN[Float](10, inputShape = Shape(3, 6))
    seq.add(rnn)
    seq.add(Dense(5))
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 5))
  }

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] =
    Array(in(0).t(), in(1).t(), in(2))

  "SimpleRNN not return sequences" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5])
        |input = np.random.random([2, 4, 5])
        |output_tensor = SimpleRNN(8, activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = SimpleRNN[Float](8, activation = "relu", inputShape = Shape(4, 5))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "SimpleRNN return sequences" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[5, 8])
        |input = np.random.random([3, 5, 8])
        |output_tensor = SimpleRNN(12, return_sequences=True)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = SimpleRNN[Float](12, returnSequences = true, inputShape = Shape(5, 8))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 5, 12))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "SimpleRNN go backwards" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 12])
        |input = np.random.random([3, 12, 12])
        |output_tensor = SimpleRNN(4, go_backwards=True, activation="sigmoid")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = SimpleRNN[Float](4, goBackwards = true,
      activation = "sigmoid", inputShape = Shape(12, 12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class SimpleRNNSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = SimpleRNN[Float](8, activation = "relu", inputShape = Shape(4, 5))
    layer.build(Shape(3, 4, 5))
    val input = Tensor[Float](3, 4, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
