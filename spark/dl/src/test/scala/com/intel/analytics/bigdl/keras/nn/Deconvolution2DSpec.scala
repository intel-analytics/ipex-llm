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
import com.intel.analytics.bigdl.nn.keras.{Deconvolution2D, Deconv2D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class Deconvolution2DSpec extends KerasBaseSpec {

  def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = {
    var w = in(0).transpose(1, 2)
    if (in.length > 1) Array(w, in(1)) // with bias
    else Array(w) // without bias
  }

  "Deconvolution2D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 12, 12])
        |input = np.random.random([8, 3, 12, 12])
        |output_tensor = Deconvolution2D(3, 3, 3, activation="relu", dim_ordering="th",
        |                                output_shape=(None, 3, 14, 14))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Deconvolution2D[Float](3, 3, 3, activation = "relu",
      inputShape = Shape(3, 12, 12))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, precision = 1e-3)
  }

  "Deconvolution2D without bias" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 12, 12])
        |input = np.random.random([32, 3, 12, 12])
        |output_tensor = Deconvolution2D(3, 3, 3, dim_ordering="th",
        |                                subsample=(2, 2), bias=False,
        |                                output_shape=(None, 3, 25, 25))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Deconv2D[Float](3, 3, 3, subsample = (2, 2), bias = false,
      inputShape = Shape(3, 12, 12))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, weightConverter, precision = 1e-3)
  }

}

class Deconvolution2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Deconvolution2D[Float](3, 3, 3, inputShape = Shape(3, 24, 24))
    layer.build(Shape(2, 3, 24, 24))
    val input = Tensor[Float](2, 3, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
