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

class Cropping2DSpec extends KerasBaseSpec {
  "Cropping2D" should "with NCHW work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5])
        |input = np.random.uniform(-1, 1, [2, 3, 4, 5])
        |output_tensor = Cropping2D(cropping=((1, 1), (1, 1)), dim_ordering='th')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = Cropping2D[Float](Array(1, 1), Array(1, 1), DataFormat.NCHW)
    checkOutputAndGrad(model, kerasCode)
  }

  "Cropping2D" should "with NHWC work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4, 5])
        |input = np.random.uniform(-1, 1, [2, 3, 4, 5])
        |output_tensor = Cropping2D(cropping=((1, 1), (1, 1)), dim_ordering='tf')(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = Cropping2D[Float](Array(1, 1), Array(1, 1), DataFormat.NHWC)
    checkOutputAndGrad(model, kerasCode)
  }

  "Cropping2D computeOutputShape NCHW" should "work properly" in {
    val layer = Cropping2D[Float](Array(2, 3), Array(2, 4))
    TestUtils.compareOutputShape(layer, Shape(3, 12, 12)) should be (true)
  }

  "Cropping2D computeOutputShape NHWC" should "work properly" in {
    val layer = Cropping2D[Float](Array(1, 3), Array(2, 2), format = DataFormat.NHWC)
    TestUtils.compareOutputShape(layer, Shape(18, 12, 3)) should be (true)
  }

}
