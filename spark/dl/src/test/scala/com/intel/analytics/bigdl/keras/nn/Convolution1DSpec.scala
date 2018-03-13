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
import com.intel.analytics.bigdl.nn.keras.{Convolution1D, Conv1D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class Convolution1DSpec extends KerasBaseSpec {

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] =
    if (in.length == 1) in // without bias
    else Array(in(0).resize(Array(1) ++ in(0).size()), in(1)) // with bias

  "Convolution1D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 20])
        |input = np.random.random([2, 12, 20])
        |output_tensor = Convolution1D(64, 3)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Convolution1D[Float](64, 3, inputShape = Shape(12, 20))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 10, 64))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "Convolution1D without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[20, 32])
        |input = np.random.random([2, 20, 32])
        |output_tensor = Convolution1D(32, 4, activation="relu", bias=False,
        |                              subsample_length=2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Conv1D[Float](32, 4, activation = "relu", subsampleLength = 2,
      bias = false, inputShape = Shape(20, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 9, 32))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class Convolution1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Convolution1D[Float](64, 3, inputShape = Shape(12, 20))
    layer.build(Shape(2, 12, 20))
    val input = Tensor[Float](2, 12, 20).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
