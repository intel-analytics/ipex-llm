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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.nn.{ResizeBilinear => BResizeBilinear}
import com.intel.analytics.zoo.pipeline.api.keras.layers.{ResizeBilinear => ZResizeBilinear}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest


class ResizeBilinearSpec extends ZooSpecHelper {

  "ResizeBilinear (3, 2) Zoo" should "be the same as BigDL" in {
    val input = Tensor[Float](1, 3, 2, 3).rand()
    val blayer = BResizeBilinear[Float](3, 2, dataFormat = DataFormat.NCHW)
    val zlayer = ZResizeBilinear[Float](3, 2, dimOrdering = "th")
    zlayer.build(Shape(-1, 3, 2, 3))
    compareOutputAndGradInput(
      blayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      zlayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      input)
  }
}

class ResizeBilinearSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = ZResizeBilinear[Float](3, 2, dimOrdering = "tf")
    layer.build(Shape(-1, 3, 2, 3))
    val input = Tensor[Float](1, 3, 2, 3).rand()

    runSerializationTest(layer, input)
  }
}
