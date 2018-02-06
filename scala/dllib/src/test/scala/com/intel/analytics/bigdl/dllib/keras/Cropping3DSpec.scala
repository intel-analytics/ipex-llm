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

package com.intel.analytics.bigdl.keras

import com.intel.analytics.bigdl.nn.{Cropping2D, _}
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.utils.{Shape, TestUtils}

class Cropping3DSpec extends KerasBaseSpec {
  "Cropping3D" should "with CHANNEL_FIRST work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5, 6])
        |input = np.random.uniform(-1, 1, [2, 3, 4, 5, 6])
        |output_tensor = Cropping3D(
        |    cropping=((1, 1), (1, 1), (1, 1)), dim_ordering='th')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = Cropping3D[Float](Array(1, 1), Array(1, 1), Array(1, 1), Cropping3D.CHANNEL_FIRST)
    checkOutputAndGrad(model, kerasCode)
  }

  "Cropping3D" should "with CHANNEL_LAST work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5, 6])
        |input = np.random.uniform(-1, 1, [2, 3, 4, 5, 6])
        |output_tensor = Cropping3D(
        |    cropping=((1, 1), (1, 1), (1, 1)), dim_ordering='tf')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = Cropping3D[Float](Array(1, 1), Array(1, 1), Array(1, 1), Cropping3D.CHANNEL_LAST)
    checkOutputAndGrad(model, kerasCode)
  }

  "Cropping3D computeOutputShape CHANNEL_FIRST" should "work properly" in {
    val layer = Cropping3D[Float](Array(2, 3), Array(2, 4), Array(1, 2))
    TestUtils.compareOutputShape(layer, Shape(3, 24, 28, 32)) should be (true)
  }

  "Cropping3D computeOutputShape CHANNEL_LAST" should "work properly" in {
    val layer = Cropping3D[Float](Array(1, 3), Array(2, 1), Array(4, 2), Cropping3D.CHANNEL_LAST)
    TestUtils.compareOutputShape(layer, Shape(32, 32, 32, 4)) should be (true)
  }

}
