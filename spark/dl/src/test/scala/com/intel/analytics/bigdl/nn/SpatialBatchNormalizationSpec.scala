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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class SpatialBatchNormalizationSpec extends FlatSpec with Matchers {

  "SpatialBatchNormalization with NHWC and NCHW format" should "have the same result" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val inputsNCHW = Tensor(2, 3, 4, 5).randn()

    val inputsNHWC = Tensor(2, 3, 4, 5).copy(inputsNCHW)
      .transpose(2, 4).transpose(2, 3).contiguous()

    val gradOutput = Tensor(2, 3, 4, 5).randn()
    val gradOutputNHWC = Tensor(2, 3, 4, 5).copy(gradOutput)
      .transpose(2, 4).transpose(2, 3).contiguous()

    val bnNCHW = SpatialBatchNormalization(3)
    val bnNHWC = SpatialBatchNormalization(3, format = DataFormat.NHWC)

    bnNHWC.weight.copy(bnNCHW.weight)
    bnNHWC.bias.copy(bnNCHW.bias)


    val output1 = bnNCHW.forward(inputsNCHW).transpose(2, 4).transpose(2, 3)
    val output2 = bnNHWC.forward(inputsNHWC)

    val gradInput1 = bnNCHW.backward(inputsNCHW, gradOutput).transpose(2, 4).transpose(2, 3)
    val gradInput2 = bnNHWC.backward(inputsNHWC, gradOutputNHWC)

    (output1.abs().sum() - output2.abs().sum()) should be < 1e-5

    (gradInput1.abs().sum() - gradInput2.abs().sum()) should be < 1e-5

    val output11 = bnNCHW.forward(inputsNCHW).transpose(2, 4).transpose(2, 3)
    val output22 = bnNHWC.forward(inputsNHWC)

    val gradInput11 = bnNCHW.backward(inputsNCHW, gradOutput).transpose(2, 4).transpose(2, 3)
    val gradInput22 = bnNHWC.backward(inputsNHWC, gradOutputNHWC)

    (output11.abs().sum() - output22.abs().sum()) should be < 1e-6

    (gradInput11.abs().sum() - gradInput22.abs().sum()) should be < 1e-6

    bnNCHW.gradWeight.sub(bnNHWC.gradWeight).abs().sum() should be < 1e-6

    bnNCHW.gradBias.sub(bnNHWC.gradBias).abs().sum() should be < 1e-6
  }
}