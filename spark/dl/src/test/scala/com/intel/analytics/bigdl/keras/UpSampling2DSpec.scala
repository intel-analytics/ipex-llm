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

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.TestUtils

class UpSampling2DSpec extends KerasBaseSpec {
  "UpSampling2D nchw" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[5, 3, 4])
        |input = np.random.uniform(-1, 1, [2, 5, 3, 4])
        |output_tensor = UpSampling2D(size=[2, 3], dim_ordering="th")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = UpSampling2D[Float](Array(2, 3))
    checkOutputAndGrad(model, kerasCode)
  }

  "UpSampling2D nhwc" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5])
        |input = np.random.uniform(-1, 1, [2, 3, 4, 5])
        |output_tensor = UpSampling2D([2, 3], dim_ordering="tf")(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = UpSampling2D[Float](Array(2, 3), DataFormat.NHWC)
    checkOutputAndGrad(model, kerasCode)
  }

  "UpSampling2D computeOutputShape NCHW" should "work properly" in {
    val layer = UpSampling2D[Float](Array(1, 2))
    TestUtils.compareOutputShape(layer, Shape(3, 4, 5)) should be (true)
  }

  "UpSampling2D computeOutputShape NHWC" should "work properly" in {
    val layer = UpSampling2D[Float](Array(3, 3), format = DataFormat.NHWC)
    TestUtils.compareOutputShape(layer, Shape(8, 12, 2)) should be (true)
  }

}
