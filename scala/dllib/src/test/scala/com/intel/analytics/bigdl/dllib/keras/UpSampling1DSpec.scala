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
import com.intel.analytics.bigdl.utils.{Shape, TestUtils}

class UpSampling1DSpec extends KerasBaseSpec {
  "UpSampling1D forward with size 1" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4])
        |input = np.random.uniform(-1, 1, [2, 3, 4])
        |output_tensor = UpSampling1D(1)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = UpSampling1D[Float](1)
    checkOutputAndGrad(model, kerasCode)
  }

  "UpSampling1D forward with size 2" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[3, 4])
        |input = np.random.uniform(-1, 1, [2, 3, 4])
        |output_tensor = UpSampling1D(2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val model = UpSampling1D[Float](2)
    checkOutputAndGrad(model, kerasCode)
  }

  "UpSampling1D computeOutputShape" should "work properly" in {
    val layer = UpSampling1D[Float](3)
    TestUtils.compareOutputShape(layer, Shape(4, 5)) should be (true)
  }

}
