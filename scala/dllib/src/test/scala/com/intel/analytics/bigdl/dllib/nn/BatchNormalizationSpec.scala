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

import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class BatchNormalizationSpec extends FlatSpec with Matchers {
  "A BatchNormalization" should "generate correct output" in {
    val bn = new BatchNormalization[Double](3)
    bn.weight(1) = 0.1
    bn.weight(2) = 0.2
    bn.weight(3) = 0.3

    bn.bias(1) = 0.1
    bn.bias(2) = 0.2
    bn.bias(3) = 0.3
    val input = Tensor[Double](3, 3)

    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output = bn.forward(input)

    output.nDimension() should be(2)
    output.size(1) should be(3)
    output.size(2) should be(3)
    output(Array(1, 1)) should be(-0.0225 +- 0.0001)
    output(Array(1, 2)) should be(-0.0449 +- 0.0001)
    output(Array(1, 3)) should be(-0.0674 +- 0.0001)
    output(Array(2, 1)) should be(0.1 +- 0.0001)
    output(Array(2, 2)) should be(0.2 +- 0.0001)
    output(Array(2, 3)) should be(0.3 +- 0.0001)
    output(Array(3, 1)) should be(0.2225 +- 0.0001)
    output(Array(3, 2)) should be(0.4449 +- 0.0001)
    output(Array(3, 3)) should be(0.6674 +- 0.0001)
  }

  "A BatchNormalization" should "generate correct gradient" in {
    val bn = new BatchNormalization[Double](3)
    bn.weight(1) = 0.1
    bn.weight(2) = 0.2
    bn.weight(3) = 0.3

    bn.bias(1) = 0.1
    bn.bias(2) = 0.2
    bn.bias(3) = 0.3
    val input = Tensor[Double](3, 3)
    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output = bn.forward(input)

    val gradOutput = Tensor[Double](3, 3)
    var j = 0.0
    gradOutput.apply1(e => {
      j += 0.1; j
    })
    val gradInput = bn.backward(input, gradOutput)

    gradInput.nDimension() should be(2)
    gradInput.size(1) should be(3)
    gradInput.size(2) should be(3)

    gradInput(Array(1, 1)) should be(-2.0412e-8 +- 1e-12)
    gradInput(Array(1, 2)) should be(-4.0825e-8 +- 1e-12)
    gradInput(Array(1, 3)) should be(-6.1237e-8 +- 1e-12)
    gradInput(Array(2, 1)) should be(-0.0 +- 0.0001)
    gradInput(Array(2, 2)) should be(-0.0 +- 0.0001)
    gradInput(Array(2, 3)) should be(-0.0 +- 0.0001)
    gradInput(Array(3, 1)) should be(2.0412e-8 +- 1e-12)
    gradInput(Array(3, 2)) should be(4.0825e-8 +- 1e-12)
    gradInput(Array(3, 3)) should be(6.1237e-8 +- 1e-12)

    bn.gradWeight.nDimension() should be(1)
    bn.gradWeight.size(1) should be(3)
    bn.gradWeight(Array(1)) should be(0.7348 +- 0.0001)
    bn.gradWeight(Array(2)) should be(0.7348 +- 0.0001)
    bn.gradWeight(Array(3)) should be(0.7348 +- 0.0001)

    bn.gradBias.nDimension() should be(1)
    bn.gradBias.size(1) should be(3)
    bn.gradBias(Array(1)) should be(1.2 +- 0.0001)
    bn.gradBias(Array(2)) should be(1.5 +- 0.0001)
    bn.gradBias(Array(3)) should be(1.8 +- 0.0001)
  }

  "A BatchNormalization evaluating" should "generate correct output" in {
    val bn = new BatchNormalization[Double](3)
    bn.weight(1) = 0.1
    bn.weight(2) = 0.2
    bn.weight(3) = 0.3

    bn.bias(1) = 0.1
    bn.bias(2) = 0.2
    bn.bias(3) = 0.3
    val input = Tensor[Double](3, 3)
    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    var output = bn.forward(input)

    val gradOutput = Tensor[Double](3, 3)
    var j = 0.0
    gradOutput.apply1(e => {
      j += 0.1; j
    })
    val gradInput = bn.backward(input, gradOutput)
    bn.evaluate()
    output = bn.forward(input)
    println(output)
    output = bn.forward(input)
    println(output)
    output = bn.forward(input)
    println(output)
    output = bn.forward(input)
    println(output)
  }

  it should "generate correct output for no batch" in {
    val bn = new BatchNormalization[Double](3)
    bn.weight(1) = 0.1
    bn.weight(2) = 0.2
    bn.weight(3) = 0.3

    bn.bias(1) = 0.1
    bn.bias(2) = 0.2
    bn.bias(3) = 0.3
    bn.evaluate()

    val input = Tensor[Double](3)
    var i = 0
    input.apply1(e => {
      i += 1; i
    })
    val output = bn.forward(input)
    output.valueAt(1) should be(0.2 +- 0.00001)
    output.valueAt(2) should be(0.6 +- 0.00001)
    output.valueAt(3) should be(1.2 +- 0.00001)
  }
}
