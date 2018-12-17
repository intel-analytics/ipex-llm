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
import com.intel.analytics.bigdl.nn.keras.{UpSampling3D, Sequential => KSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

import scala.util.Random

class UpSampling3DSpec extends KerasBaseSpec {

  "UpSampling3D with default size" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 8, 10, 12])
        |input = np.random.random([2, 3, 8, 10, 12])
        |output_tensor = UpSampling3D(dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = UpSampling3D[Float](inputShape = Shape(3, 8, 10, 12))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

  "UpSampling3D with different sizes" should "be the same as Keras" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2, 12, 12, 12])
        |input = np.random.random([2, 2, 12, 12, 12])
        |output_tensor = UpSampling3D(size=(2, 1, 3), dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val seq = KSequential[Float]()
    val layer = UpSampling3D[Float](size = (2, 1, 3), inputShape = Shape(2, 12, 12, 12))
    seq.add(layer)
    checkOutputAndGrad(seq.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      kerasCode)
  }

}

class UpSampling3DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = UpSampling3D[Float](inputShape = Shape(3, 8, 10, 12))
    layer.build(Shape(2, 3, 8, 10, 12))
    val input = Tensor[Float](2, 3, 8, 10, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
