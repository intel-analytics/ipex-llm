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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.nn.{CategoricalCrossEntropy, SpatialSeparableConvolution}

class SpatialSeparableConvolutionSpec extends KerasBaseSpec {
  "SpatialSeperableConvolution" should "be ok" in {
    ifskipTest()
    val kerasCode =
      """
        |input_tensor = Input(shape=[5, 5, 2])
        |output_tensor = SeparableConv2D(2, 2, 2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
        |input = np.random.uniform(0, 1, [2, 5, 5, 2])
      """.stripMargin
    val layer = SpatialSeparableConvolution[Float](2, 2, 1, 2, 2, dataFormat = DataFormat.NHWC)
    checkOutputAndGrad(layer, kerasCode)
  }

  "SpatialSeperableConvolution" should "be ok when depth multipler is not 1" in {
    ifskipTest()
    val kerasCode =
      """
        |input_tensor = Input(shape=[5, 5, 2])
        |output_tensor = SeparableConv2D(4, 2, 2, depth_multiplier=2)(input_tensor)
        |model = Model(input=input_tensor, output=output_tensor)
        |input = np.random.uniform(0, 1, [2, 5, 5, 2])
      """.stripMargin
    val layer = SpatialSeparableConvolution[Float](2, 4, 2, 2, 2, dataFormat = DataFormat.NHWC)
    checkOutputAndGrad(layer, kerasCode)
  }
}
