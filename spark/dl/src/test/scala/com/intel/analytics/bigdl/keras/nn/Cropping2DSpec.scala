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
import com.intel.analytics.bigdl.nn.keras.{Cropping2D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class Cropping2DSpec extends KerasBaseSpec {

  "Cropping2D NCHW" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 8, 12])
        |input = np.random.random([2, 3, 8, 12])
        |output_tensor = Cropping2D(((1, 2), (3, 1)), dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Cropping2D[Float](((1, 2), (3, 1)), inputShape = Shape(3, 8, 12))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "Cropping2D NHWC" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 5, 3])
        |input = np.random.random([2, 4, 5, 3])
        |output_tensor = Cropping2D(((0, 1), (1, 1)), dim_ordering="tf")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = Cropping2D[Float](((0, 1), (1, 1)), dimOrdering = "tf",
      inputShape = Shape(4, 5, 3))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class Cropping2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Cropping2D[Float](inputShape = Shape(3, 8, 12))
    layer.build(Shape(2, 3, 8, 12))
    val input = Tensor[Float](2, 3, 8, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
