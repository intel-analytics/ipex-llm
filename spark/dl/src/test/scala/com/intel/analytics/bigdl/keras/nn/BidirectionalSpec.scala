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
import com.intel.analytics.bigdl.nn.keras.{Bidirectional, LSTM, SimpleRNN, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class BidirectionalSpec extends KerasBaseSpec {

  "Bidirectional SimpleRNN concat" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[8, 12])
        |input = np.random.random([3, 8, 12])
        |output_tensor = Bidirectional(SimpleRNN(4, return_sequences=True))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Bidirectional[Float](SimpleRNN(4, returnSequences = true),
      inputShape = Shape(8, 12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 8, 8))
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] =
      Array(in(0).t(), in(1).t(), in(2), in(3).t(), in(4).t(), in(5))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "Bidirectional LSTM sum" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[32, 32])
        |input = np.random.random([3, 32, 32])
        |output_tensor = Bidirectional(LSTM(12, return_sequences=True),
        |                              merge_mode="sum")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Bidirectional[Float](LSTM(12, returnSequences = true),
      mergeMode = "sum", inputShape = Shape(32, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 32, 12))
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
      val w1 = Tensor[Float](in(0).size(2)*4, in(0).size(1))
      val w2 = Tensor[Float](in(2).size(1)*4)
      val w3 = Tensor[Float](in(1).size(2)*4, in(1).size(1))
      val w4 = w1.clone()
      val w5 = w2.clone()
      val w6 = w3.clone()
      var i = 0
      while(i < 4) {
        w1.narrow(1, 1 + i * in(0).size(2), in(0).size(2)).copy(in(3*i).t())
        w2.narrow(1, 1 + i * in(2).size(1), in(2).size(1)).copy(in(2 + 3*i))
        w3.narrow(1, 1 + i * in(1).size(2), in(1).size(2)).copy(in(1 + 3*i).t())
        w4.narrow(1, 1 + i * in(0).size(2), in(0).size(2)).copy(in(3*i + 12).t())
        w5.narrow(1, 1 + i * in(2).size(1), in(2).size(1)).copy(in(2 + 3*i + 12))
        w6.narrow(1, 1 + i * in(1).size(2), in(1).size(2)).copy(in(1 + 3*i + 12).t())
        i += 1
      }
      Array(w1, w2, w3, w4, w5, w6)
    }
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class BidirectionalSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Bidirectional[Float](SimpleRNN(4, returnSequences = true),
      inputShape = Shape(8, 12))
    layer.build(Shape(3, 8, 12))
    val input = Tensor[Float](3, 8, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
