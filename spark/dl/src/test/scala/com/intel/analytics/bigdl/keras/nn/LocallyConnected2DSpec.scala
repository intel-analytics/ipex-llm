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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.nn.keras.{LocallyConnected2D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class LocallyConnected2DSpec extends KerasBaseSpec {

  def weightConverter(data: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    val out = new Array[Tensor[Float]](data.length)
    val d1l: Int = data(0).size(1)
    val d2l: Int = data(0).size(2)
    val d3l: Int = data(0).size(3)
    out(0) = Tensor(d1l, d3l, d2l)
    val page: Int = d2l * d3l
    for (i <- 0 to d1l * d2l * d3l - 1) {
      val d1 = i / page + 1
      val d2 = (i % page) / (d3l) + 1
      val d3 = (i % page) % d3l + 1
      val v = data(0).valueAt(d1, d2, d3)
      out(0).setValue(d1, d3, d2, v)
    }
    if (data.length > 1) {
      out(1) = data(1)
    }
    out
  }

  "LocallyConnected2D NCHW" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 24, 24])
        |input = np.random.random([2, 12, 24, 24])
        |output_tensor = LocallyConnected2D(32, 2, 2, dim_ordering="th",
        |                                   activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = LocallyConnected2D[Float](32, 2, 2,
      activation = "relu", inputShape = Shape(12, 24, 24))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "LocallyConnected2D without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[8, 32, 32])
        |input = np.random.random([2, 8, 32, 32])
        |output_tensor = LocallyConnected2D(64, 3, 3, bias=False,
        |                                   dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = LocallyConnected2D[Float](64, 3, 3, bias = false, inputShape = Shape(8, 32, 32))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "LocallyConnected2D NHWC" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[24, 24, 12])
        |input = np.random.random([2, 24, 24, 12])
        |output_tensor = LocallyConnected2D(32, 2, 2, dim_ordering="tf",
        |                                   activation="relu")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = LocallyConnected2D[Float](32, 2, 2, activation = "relu",
      dimOrdering = "tf", inputShape = Shape(24, 24, 12))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class LocallyConnected2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = LocallyConnected2D[Float](32, 2, 2, activation = "relu",
      inputShape = Shape(12, 24, 24))
    layer.build(Shape(2, 12, 24, 24))
    val input = Tensor[Float](2, 12, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
