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

package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.keras.KerasBaseSpec
import com.intel.analytics.bigdl.tensor.Tensor

class HighwaySpec extends KerasBaseSpec {
  "highway forward backward" should "work properly" in {
    val kerasCode =
      """
        |input_tensor = Input(shape=[2])
        |input = np.random.uniform(0, 1, [3, 2])
        |output_tensor = Highway()(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
      """.stripMargin
    val highway = new Highway[Float](2)
    def weightConverter(in: Array[Tensor[Float]]): Array[Tensor[Float]] = Array(in(0).t(), in(1))
    checkOutputAndGrad(highway, kerasCode, weightConverter)
  }
}
