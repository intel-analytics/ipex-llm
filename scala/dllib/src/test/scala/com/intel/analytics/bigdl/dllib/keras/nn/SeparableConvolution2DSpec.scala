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
import com.intel.analytics.bigdl.nn.keras.{SeparableConvolution2D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class SeparableConvolution2DSpec extends KerasBaseSpec {

  "SeparableConvolution2D NCHW" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5])
        |input = np.random.random([2, 3, 4, 5])
        |output_tensor = SeparableConvolution2D(3, 3, 3, dim_ordering='th')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = SeparableConvolution2D[Float](3, 3, 3, inputShape = Shape(3, 4, 5))
    seq.add(layer)
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
      if (in.length == 2) {
        val bias = if (layer.dimOrdering == DataFormat.NCHW) in(1).size(1)
        else in(1).size(4)
        val out = Tensor[Float](bias)
        Array(in(0), in(1), out)
      }
      else in
    }
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "SeparableConvolution2D without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5])
        |input = np.random.random([2, 3, 4, 5])
        |output_tensor = SeparableConvolution2D(3, 3, 3, dim_ordering='th',
        |                                       bias=False)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = SeparableConvolution2D[Float](3, 3, 3, bias = false, inputShape = Shape(3, 4, 5))
    seq.add(layer)
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
      if (in.length == 2) {
        val bias = if (layer.dimOrdering == DataFormat.NCHW) in(1).size(1)
        else in(1).size(4)
        val out = Tensor[Float](bias)
        Array(in(0), in(1), out)
      }
      else in
    }
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

  "SeparableConvolution2D NHWC" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 12, 3])
        |input = np.random.random([2, 12, 12, 3])
        |output_tensor = SeparableConvolution2D(8, 2, 2, activation="relu",
        |                                       dim_ordering='tf')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = SeparableConvolution2D[Float](8, 2, 2, activation = "relu",
      dimOrdering = "tf", inputShape = Shape(12, 12, 3))
    seq.add(layer)
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
      if (in.length == 2) {
        val bias = if (layer.dimOrdering == DataFormat.NCHW) in(1).size(1)
        else in(1).size(4)
        val out = Tensor[Float](bias)
        Array(in(0), in(1), out)
      }
      else in
    }
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter)
  }

}

class SeparableConvolution2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = SeparableConvolution2D[Float](1, 2, 2, inputShape = Shape(3, 128, 128))
    layer.build(Shape(2, 3, 128, 128))
    val input = Tensor[Float](2, 3, 128, 128).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
