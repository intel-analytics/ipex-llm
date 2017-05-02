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

package com.intel.analytics.bigdl.nn.fixpoint

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat

@com.intel.analytics.bigdl.tags.Parallel
class SpatialConvolutionSpec extends FlatSpec with Matchers {
  "A SpatialConvolution layer" should "generate correct output" in {
    val nInputPlane = 1
    val nOutputPlane = 1
    val kW = 2
    val kH = 2
    val dW = 1
    val dH = 1
    val padW = 0
    val padH = 0
    val layer = new SpatialConvolution[Float](nInputPlane, nOutputPlane,
      kW, kH, dW, dH, padW, padH)

    val inputData = Array(
      1.0f, 2f, 3f,
      4f, 5f, 6f,
      7f, 8f, 9f
    )

    val kernelData = Array(
      2.0f, 3f,
      4f, 5f
    )

    val biasData = Array(0.0f)

    layer.weight.copy(Tensor(Storage(kernelData), 1, Array(nOutputPlane,
      nInputPlane, kH, kW)))
    layer.bias.copy(Tensor(Storage(biasData), 1, Array(nOutputPlane)))
    val input = Tensor(Storage(inputData), 1, Array(1, 3, 3))
    val output = layer.updateOutput(input)
    println(output)
    output(Array(1, 1, 1)) should be(49)
    output(Array(1, 1, 2)) should be(63)
    output(Array(1, 2, 1)) should be(91)
    output(Array(1, 2, 2)) should be(105)
  }
}
