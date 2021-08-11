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
import com.intel.analytics.bigdl.nn.keras.{GRU, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class GRUSpec extends KerasBaseSpec {

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    val w1 = Tensor[Float](in(0).size(2)*3, in(0).size(1))
    val w2 = Tensor[Float](in(2).size(1)*3)
    val w3 = Tensor[Float](in(1).size(2)*2, in(1).size(1))
    w1.narrow(1, 1, in(0).size(2)).copy(in(3).t())
    w1.narrow(1, 1 + in(0).size(2), in(0).size(2)).copy(in(0).t())
    w1.narrow(1, 1 + 2*in(0).size(2), in(0).size(2)).copy(in(6).t())
    w2.narrow(1, 1, in(2).size(1)).copy(in(5))
    w2.narrow(1, 1 + in(2).size(1), in(2).size(1)).copy(in(2))
    w2.narrow(1, 1 + 2*in(2).size(1), in(2).size(1)).copy(in(8))
    w3.narrow(1, 1, in(1).size(2)).copy(in(4).t())
    w3.narrow(1, 1 + in(1).size(2), in(1).size(2)).copy(in(1).t())
    Array(w1, w2, w3, in(7).t())
  }

  "GRU not return sequences" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[28, 28])
        |input = np.random.random([2, 28, 28])
        |output_tensor = GRU(128)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = GRU[Float](128, inputShape = Shape(28, 28))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 128))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "GRU return sequences" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[32, 32])
        |input = np.random.random([2, 32, 32])
        |output_tensor = GRU(36, return_sequences=True, activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = GRU[Float](36, returnSequences = true,
      activation = "relu", inputShape = Shape(32, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 32, 36))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "GRU go backwards and return sequences" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[28, 32])
        |input = np.random.random([1, 28, 32])
        |output_tensor = GRU(16, return_sequences=True, go_backwards=True)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = GRU[Float](16, returnSequences = true,
      goBackwards = true, inputShape = Shape(28, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 28, 16))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class GRUSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = GRU[Float](16, returnSequences = true,
      goBackwards = true, inputShape = Shape(28, 32))
    layer.build(Shape(2, 28, 32))
    val input = Tensor[Float](2, 28, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
