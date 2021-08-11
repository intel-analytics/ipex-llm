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
import com.intel.analytics.bigdl.nn.keras.{UpSampling2D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class UpSampling2DSpec extends KerasBaseSpec {

  "UpSampling2D NCHW" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[4, 8, 8])
        |input = np.random.random([2, 4, 8, 8])
        |output_tensor = UpSampling2D(dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = UpSampling2D[Float](inputShape = Shape(4, 8, 8))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "UpSampling2D NHWC" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[12, 14, 3])
        |input = np.random.random([1, 12, 14, 3])
        |output_tensor = UpSampling2D(size=(1, 3), dim_ordering="tf")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = UpSampling2D[Float](size = (1, 3), dimOrdering = "tf",
      inputShape = Shape(12, 14, 3))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class UpSampling2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = UpSampling2D[Float](inputShape = Shape(4, 8, 8))
    layer.build(Shape(2, 4, 8, 8))
    val input = Tensor[Float](2, 4, 8, 8).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
