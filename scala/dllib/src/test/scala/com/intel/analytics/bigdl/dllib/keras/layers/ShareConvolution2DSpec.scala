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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.nn.{ReLU, SpatialShareConvolution, Sequential => BSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class ShareConvolution2DSpec extends ZooSpecHelper {

  "ShareConvolution2D Zoo" should "be the same as BigDL" in {
    val blayer = SpatialShareConvolution[Float](4, 12, 2, 3)
    val zlayer = ShareConvolution2D[Float](12, 3, 2, inputShape = Shape(4, 32, 32))
    zlayer.build(Shape(-1, 4, 32, 32))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 12, 30, 31))
    val input = Tensor[Float](Array(2, 4, 32, 32)).rand()
    compareOutputAndGradInputSetWeights(blayer, zlayer, input)
  }

  "ShareConv2D Zoo with activation" should "be the same as BigDL" in {
    val bmodel = BSequential[Float]()
      .add(SpatialShareConvolution(3, 64, 7, 7, 2, 1, 2, 3))
      .add(ReLU())
    val zlayer = ShareConv2D[Float](64, 7, 7, activation = "relu", subsample = (1, 2),
      padH = 3, padW = 2, inputShape = Shape(3, 24, 32))
    zlayer.build(Shape(-1, 3, 24, 32))
    zlayer.getOutputShape().toSingle().toArray should be (Array(-1, 64, 24, 15))
    val input = Tensor[Float](Array(2, 3, 24, 32)).rand()
    compareOutputAndGradInputSetWeights(
      bmodel.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]], zlayer, input)
  }

}

class ShareConvolution2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ShareConv2D[Float](64, 7, 7, activation = "relu", subsample = (1, 2),
      padH = 3, padW = 2, inputShape = Shape(3, 24, 32))
    layer.build(Shape(2, 3, 24, 32))
    val input = Tensor[Float](2, 3, 24, 32).rand()
    runSerializationTest(layer, input)
  }
}
