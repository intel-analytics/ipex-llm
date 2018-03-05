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
import com.intel.analytics.bigdl.nn.keras.{AtrousConvolution1D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class AtrousConvolution1DSpec extends KerasBaseSpec {

  "AtrousConvolution1D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[8, 32])
        |input = np.random.random([2, 8, 32])
        |output_tensor = AtrousConvolution1D(64, 3, activation="tanh",
        |                                    atrous_rate=2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = AtrousConvolution1D[Float](64, 3, activation = "tanh",
      atrousRate = 2, inputShape = Shape(8, 32))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4, 64))
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
      var w = in(0).transpose(1, 4)
      w = w.transpose(2, 3)
      w = w.transpose(3, 4)
      Array(w, in(1))
    }
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class AtrousConvolution1DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = AtrousConvolution1D[Float](64, 3, inputShape = Shape(8, 32))
    layer.build(Shape(2, 8, 32))
    val input = Tensor[Float](2, 8, 32).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
