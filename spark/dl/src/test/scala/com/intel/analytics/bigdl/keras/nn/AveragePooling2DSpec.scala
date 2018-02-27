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
import com.intel.analytics.bigdl.nn.keras.{AveragePooling2D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class AveragePooling2DSpec extends KerasBaseSpec {

  "AveragePooling2D NCHW" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 24, 24])
        |input = np.random.random([2, 3, 24, 24])
        |output_tensor = AveragePooling2D(dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = AveragePooling2D[Float](inputShape = Shape(3, 24, 24))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 12, 12))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "AveragePooling2D NHWC" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[20, 32, 4])
        |input = np.random.random([2, 20, 32, 4])
        |output_tensor = AveragePooling2D(pool_size=(2, 3), strides=(1, 2),
        |                                 border_mode="same", dim_ordering="tf")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = AveragePooling2D[Float](poolSize = (2, 3), strides = (1, 2),
      borderMode = "same", dimOrdering = "tf", inputShape = Shape(20, 32, 4))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 20, 16, 4))
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class AveragePooling2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = AveragePooling2D[Float](inputShape = Shape(3, 24, 24))
    layer.build(Shape(2, 3, 24, 24))
    val input = Tensor[Float](2, 3, 24, 24).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
