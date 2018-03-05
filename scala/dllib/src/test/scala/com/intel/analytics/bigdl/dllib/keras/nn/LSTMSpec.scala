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
import com.intel.analytics.bigdl.nn.keras.{LSTM, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class LSTMSpec extends KerasBaseSpec {

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    val w1 = Tensor[Float](in(0).size(2)*4, in(0).size(1))
    val w2 = Tensor[Float](in(2).size(1)*4)
    val w3 = Tensor[Float](in(1).size(2)*4, in(1).size(1))
    var i = 0
    while(i < 4) {
      w1.narrow(1, 1 + i * in(0).size(2), in(0).size(2)).copy(in(3*i).t())
      w2.narrow(1, 1 + i * in(2).size(1), in(2).size(1)).copy(in(2 + 3*i))
      w3.narrow(1, 1 + i * in(1).size(2), in(1).size(2)).copy(in(1 + 3*i).t())
      i += 1
    }
    Array(w1, w2, w3)
  }

  "LSTM not return sequences" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[10, 12])
        |input = np.random.random([3, 10, 12])
        |output_tensor = LSTM(32)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = LSTM[Float](32, inputShape = Shape(10, 12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 32))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "LSTM return sequences" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[32, 32])
        |input = np.random.random([2, 32, 32])
        |output_tensor = LSTM(8, return_sequences=True, inner_activation="sigmoid")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = LSTM[Float](8, returnSequences = true,
      innerActivation = "sigmoid", inputShape = Shape(32, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 32, 8))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "LSTM go backwards and return sequences" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[28, 32])
        |input = np.random.random([1, 28, 32])
        |output_tensor = LSTM(10, return_sequences=True, go_backwards=True)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = LSTM[Float](10, returnSequences = true,
      goBackwards = true, inputShape = Shape(28, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 28, 10))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class LSTMSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = LSTM[Float](8, returnSequences = true,
      innerActivation = "sigmoid", inputShape = Shape(32, 32))
    layer.build(Shape(3, 32, 32))
    val input = Tensor[Float](3, 32, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
