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
import com.intel.analytics.bigdl.nn.keras.{MaxoutDense, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class MaxoutDenseSpec extends KerasBaseSpec {

  def weightConverter(data: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    val out = new Array[Tensor[Float]](data.length)
    out(0) = Tensor(12, 32)
    val weight = out.head.storage().array()
    var index = 0
    for (i <- 1 to 4) {
      val sliceW = data(0).select(1, i).t.clone().storage().array()
      System.arraycopy(sliceW, 0, weight, index, sliceW.size)
      index += sliceW.size
    }
    if (data.length > 1) {
      out(1) = data(1)
    }
    out
  }

  "MaxoutDense" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12])
        |input = np.random.random([4, 12])
        |output_tensor = MaxoutDense(8)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = MaxoutDense[Float](8, inputShape = Shape(12))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "MaxoutDense without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12])
        |input = np.random.random([4, 12])
        |output_tensor = MaxoutDense(8, bias=False)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = MaxoutDense[Float](8, bias = false, inputShape = Shape(12))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class MaxoutDenseSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = MaxoutDense[Float](8, inputShape = Shape(12))
    layer.build(Shape(3, 12))
    val input = Tensor[Float](3, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
