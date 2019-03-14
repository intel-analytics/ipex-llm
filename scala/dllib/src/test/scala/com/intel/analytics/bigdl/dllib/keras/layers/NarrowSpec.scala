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

import com.intel.analytics.bigdl.nn.{Narrow => BNarrow}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{Narrow => ZNarrow}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.autograd.Parameter
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import com.intel.analytics.zoo.pipeline.api.keras.models.{Model, Sequential}

import scala.util.Random

class NarrowSpec extends ZooSpecHelper {

  "Narrow Zoo 1D" should "be the same as BigDL" in {
    val blayer = BNarrow[Float](1, 1, 2)
    val input1 = Parameter[Float](inputShape = Shape(4), name = "input1")
    val zlayer = new ZNarrow[Float](0, 0, 2).from(input1)
    val model = Model(input1, zlayer)
    model.getOutputShape().toSingle().toArray should be (Array(2))
    val input = Tensor[Float](Array(4)).rand()
    val output = model.forward(input).toTensor[Float]
    output.size() should be (Array(2))
    output.toTensor[Float].almostEqual(blayer.forward(input), 1e-4)
  }

  "Narrow Zoo 1D with batch" should "be the same as BigDL" in {
    val blayer = BNarrow[Float](1, 1, 2)
    val zlayer = ZNarrow[Float](0, 0, 2, inputShape = Shape(3))
    zlayer.build(Shape(-1, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(2, 3))
    val input = Tensor[Float](Array(3, 3)).rand()
    val output = zlayer.forward(input).toTensor[Float]
    output.size() should be (Array(2, 3))
    output.toTensor[Float].almostEqual(blayer.forward(input), 1e-4)
  }

  "Narrow Zoo 2D" should "be the same as BigDL" in {
    val blayer = BNarrow[Float](2, 3, -1)
    val zlayer = ZNarrow[Float](1, 2, -1, inputShape = Shape(3))
    zlayer.build(Shape(-1, 3))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 1))
    val input = Tensor[Float](Array(2, 3)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "Narrow Zoo 3D" should "be the same as BigDL" in {
    val blayer = BNarrow[Float](2, 2)
    val zlayer = ZNarrow[Float](1, 1, inputShape = Shape(5, 6))
    zlayer.build(Shape(-1, 5, 6))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 1, 6))
    val input = Tensor[Float](Array(4, 5, 6)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "Narrow Zoo 3D with negative length" should "be the same as BigDL" in {
    val blayer = BNarrow[Float](3, 4, -1)
    val zlayer = ZNarrow[Float](2, 3, -1, inputShape = Shape(5, 6))
    zlayer.build(Shape(-1, 5, 6))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 5, 3))
    val input = Tensor[Float](Array(4, 5, 6)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "Narrow Zoo 4D" should "be the same as BigDL" in {
    val blayer = BNarrow[Float](2, 3, 3)
    val zlayer = ZNarrow[Float](1, 2, 3, inputShape = Shape(8, 5, 6))
    zlayer.build(Shape(-1, 8, 5, 6))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 3, 5, 6))
    val input = Tensor[Float](Array(2, 8, 5, 6)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

  "Narrow Zoo 4D with negative length" should "be the same as BigDL" in {
    val blayer = BNarrow[Float](-1, 4, -2)
    val zlayer = ZNarrow[Float](-1, 3, -2, inputShape = Shape(5, 6, 7))
    zlayer.build(Shape(-1, 5, 6, 7))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 5, 6, 3))
    val input = Tensor[Float](Array(2, 5, 6, 7)).rand()
    compareOutputAndGradInput(blayer, zlayer, input)
  }

// Open the restriction for batch dimension for now.
//  "Narrow the batch dimension" should "raise an exception" in {
//    intercept[RuntimeException] {
//      val zlayer = ZNarrow[Float](0, 0, inputShape = Shape(2, 3, 4))
//      zlayer.build(Shape(-1, 2, 3, 4))
//    }
//  }

  "Narrow offset too large" should "raise an exception" in {
    intercept[RuntimeException] {
      val zlayer = ZNarrow[Float](1, 2, inputShape = Shape(2, 3, 4))
      zlayer.build(Shape(-1, 2, 3, 4))
    }
  }

  "Narrow length too large" should "raise an exception" in {
    intercept[RuntimeException] {
      val zlayer = ZNarrow[Float](1, 1, 2, inputShape = Shape(2, 3, 4))
      zlayer.build(Shape(-1, 2, 3, 4))
    }
  }

}

class NarrowSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZNarrow[Float](1, 1, inputShape = Shape(5, 6))
    layer.build(Shape(2, 5, 6))
    val input = Tensor[Float](2, 5, 6).apply1(_ => Random.nextFloat())
    runSerializationTest(layer, input)
  }
}
