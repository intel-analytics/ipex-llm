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

import com.intel.analytics.bigdl.nn.ConvLSTMPeephole3D
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest

import scala.util.Random

class ConvLSTM3DSpec extends ZooSpecHelper {

  "ConvLSTM3D" should "forward and backward properly with correct output shape" in {
    val layer = ConvLSTM3D[Float](10, 3, inputShape = Shape(5, 4, 8, 10, 12), borderMode = "same")
    layer.build(Shape(-1, 5, 4, 8, 10, 12))
    val input = Tensor[Float](Array(3, 5, 4, 8, 10, 12)).rand()
    val output = layer.forward(input)
    val expectedOutputShape = layer.getOutputShape().toSingle().toArray
    val actualOutputShape = output.size()
    require(expectedOutputShape.drop(1).sameElements(actualOutputShape.drop(1)))
    val gradInput = layer.backward(input, output)
  }

  "ConvLSTM3D return sequences and go backwards" should "forward and backward " +
    "properly with correct output shape" in {
    val layer = ConvLSTM3D[Float](12, 4, returnSequences = true, goBackwards = true,
      borderMode = "same", inputShape = Shape(20, 3, 12, 12, 12))
    layer.build(Shape(-1, 20, 3, 12, 12, 12))
    val input = Tensor[Float](Array(4, 20, 3, 12, 12, 12)).rand()
    val output = layer.forward(input)
    val expectedOutputShape = layer.getOutputShape().toSingle().toArray
    val actualOutputShape = output.size()
    require(expectedOutputShape.drop(1).sameElements(actualOutputShape.drop(1)))
    val gradInput = layer.backward(input, output)
  }

  "ConvLSTM3D with same padding" should "be the same as BigDL" in {
    val blayer = com.intel.analytics.bigdl.nn.Recurrent[Float]()
      .add(ConvLSTMPeephole3D[Float](4, 4, 2, 2, withPeephole = false))
    val zlayer = ConvLSTM3D[Float](4, 2, returnSequences = true, borderMode = "same",
      inputShape = Shape(12, 4, 8, 8, 8))
    zlayer.build(Shape(-1, 12, 4, 8, 8, 8))
    val input = Tensor[Float](Array(4, 12, 4, 8, 8, 8)).rand()
    compareOutputAndGradInputSetWeights(blayer, zlayer, input)
  }

  "ConvLSTM3D with valid padding" should "work" in {
    val zlayer = ConvLSTM3D[Float](4, 2, returnSequences = true,
      borderMode = "valid", inputShape = Shape(12, 4, 8, 8, 8))
    zlayer.build(Shape(-1, 12, 4, 8, 8, 8))
    val input = Tensor[Float](Array(4, 12, 4, 8, 8, 8)).rand()
    zlayer.forward(input)
  }
}

class ConvLSTM3DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ConvLSTM3D[Float](10, 3, inputShape = Shape(5, 4, 8, 10, 12))
    layer.build(Shape(2, 5, 4, 8, 10, 12))
    val input = Tensor[Float](2, 5, 4, 8, 10, 12).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
