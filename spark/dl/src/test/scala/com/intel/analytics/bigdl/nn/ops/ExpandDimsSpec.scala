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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.nn.{Unsqueeze => ExpandDims}
import org.scalatest.{FlatSpec, Matchers}

class ExpandDimsSpec extends FlatSpec with Matchers {
  "ExpandDims operation Float" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input = Tensor[Float](Array(2, 3, 5))

    val expectOutput1 = input.clone().resize(Array(1, 2, 3, 5))
    val expectOutput2 = input.clone().resize(Array(2, 3, 1, 5))
    val expectOutput3 = input.clone().resize(Array(2, 3, 5, 1))

    val output1 = ExpandDims(1).forward(input)
    val output2 = ExpandDims(3).forward(input)
    val output3 = ExpandDims(4).forward(input)

    output1 should be(expectOutput1)
    output2 should be(expectOutput2)
    output3 should be(expectOutput3)
  }

  "ExpandDims operation Int" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input = Tensor[Int](Array(2, 3, 5))

    val expectOutput1 = input.clone().resize(Array(1, 2, 3, 5))
    val expectOutput2 = input.clone().resize(Array(2, 3, 1, 5))
    val expectOutput3 = input.clone().resize(Array(2, 3, 5, 1))

    val output1 = ExpandDims(1).forward(input)
    val output2 = ExpandDims(3).forward(input)
    val output3 = ExpandDims(4).forward(input)

    output1 should be(expectOutput1)
    output2 should be(expectOutput2)
    output3 should be(expectOutput3)
  }

  "ExpandDims operation Double" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input = Tensor[Double](Array(2, 3, 5))

    val expectOutput1 = input.clone().resize(Array(1, 2, 3, 5))
    val expectOutput2 = input.clone().resize(Array(2, 3, 1, 5))
    val expectOutput3 = input.clone().resize(Array(2, 3, 5, 1))

    val output1 = ExpandDims(1).forward(input)
    val output2 = ExpandDims(3).forward(input)
    val output3 = ExpandDims(4).forward(input)

    output1 should be(expectOutput1)
    output2 should be(expectOutput2)
    output3 should be(expectOutput3)
  }

  "ExpandDims operation Short" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input = Tensor[Short](Array(2, 3, 5))

    val expectOutput1 = input.clone().resize(Array(1, 2, 3, 5))
    val expectOutput2 = input.clone().resize(Array(2, 3, 1, 5))
    val expectOutput3 = input.clone().resize(Array(2, 3, 5, 1))

    val output1 = ExpandDims(1).forward(input)
    val output2 = ExpandDims(3).forward(input)
    val output3 = ExpandDims(4).forward(input)

    output1 should be(expectOutput1)
    output2 should be(expectOutput2)
    output3 should be(expectOutput3)
  }

  "ExpandDims operation Long" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input = Tensor[Long](Array(2, 3, 5))

    val expectOutput1 = input.clone().resize(Array(1, 2, 3, 5))
    val expectOutput2 = input.clone().resize(Array(2, 3, 1, 5))
    val expectOutput3 = input.clone().resize(Array(2, 3, 5, 1))

    val output1 = ExpandDims(1).forward(input)
    val output2 = ExpandDims(3).forward(input)
    val output3 = ExpandDims(4).forward(input)

    output1 should be(expectOutput1)
    output2 should be(expectOutput2)
    output3 should be(expectOutput3)
  }
}
