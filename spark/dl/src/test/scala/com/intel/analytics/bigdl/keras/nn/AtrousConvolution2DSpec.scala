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
import com.intel.analytics.bigdl.nn.keras.{AtrousConvolution2D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class AtrousConvolution2DSpec extends KerasBaseSpec {

  "AtrousConvolution2D" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 64, 64])
        |input = np.random.random([2, 3, 64, 64])
        |output_tensor = AtrousConvolution2D(32, 2, 4, activation="sigmoid",
        |                                    dim_ordering="th",
        |                                    atrous_rate=(2,2))(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = AtrousConvolution2D[Float](32, 2, 4, activation = "sigmoid",
      atrousRate = (2, 2), inputShape = Shape(3, 64, 64))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode, precision = 1e-2)
  }

}

class AtrousConvolution2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = AtrousConvolution2D[Float](32, 2, 4, atrousRate = (2, 2),
      inputShape = Shape(3, 64, 64))
    layer.build(Shape(2, 3, 64, 64))
    val input = Tensor[Float](2, 3, 64, 64).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
