/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import java.lang.reflect.InvocationTargetException

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.models.Sequential
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class BatchNormalizationSpec extends KerasBaseSpec {

  // Compared results with Keras on Python side
  "BatchNormalization" should "work properly for 4D input" in {
    val seq = Sequential[Float]()
    val layer = BatchNormalization[Float](betaInit = "glorot_uniform",
      gammaInit = "normal", inputShape = Shape(3, 12, 12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 3, 12, 12))
    val input = Tensor[Float](2, 3, 12, 12).rand()
    val output = seq.forward(input)
    val gradInput = seq.backward(input, output)
  }

  "BatchNormalization" should "work properly for 2D input" in {
    val seq = Sequential[Float]()
    val layer = BatchNormalization[Float](betaInit = "glorot_uniform",
      gammaInit = "normal", inputShape = Shape(12))
    seq.add(layer)
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 12))
    val input = Tensor[Float](2, 12).rand()
    val output = seq.forward(input)
    val gradInput = seq.backward(input, output)
  }

  "BatchNormalization" should "not work properly for 3D inputMeanSquaredLogarithmicErrorSpec:" in {
    val thrown = intercept[InvocationTargetException] {
      val seq = Sequential[Float]()
      val layer = BatchNormalization[Float](betaInit = "glorot_uniform",
        gammaInit = "normal", inputShape = Shape(3, 12))
      seq.add(layer)
      seq.getOutputShape().toSingle().toArray should be(Array(-1, 3, 12))
      val input = Tensor[Float](2, 3, 12).rand()
      val output = seq.forward(input)
      val gradInput = seq.backward(input, output)
    }
    assert(thrown.getTargetException.getMessage()
      .contains("BatchNormalization requires 4D or 2D input"))
  }

}

class BatchNormalizationSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = BatchNormalization[Float](inputShape = Shape(3, 12, 12))
    layer.build(Shape(2, 3, 12, 12))
    val input = Tensor[Float](2, 3, 12, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
