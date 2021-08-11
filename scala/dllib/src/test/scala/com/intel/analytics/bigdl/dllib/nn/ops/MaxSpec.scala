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
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import com.intel.analytics.bigdl.utils.{RandomGenerator, T}
import org.scalatest.{FlatSpec, Matchers}

class MaxSpec extends FlatSpec with Matchers {
  "Max operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    RandomGenerator.RNG.setSeed(10)
    val input =
      T(
        Tensor.range(1, 10).resize(2, 5),
        Tensor.scalar[Int](1)
      )

    val expectOutput = Tensor(T(5f, 10f))

    val output = Max(startFromZero = true).forward(input)
    output should be(expectOutput)
  }

  "Max operation forward one-element tensor index" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    RandomGenerator.RNG.setSeed(10)
    val input =
      T(
        Tensor.range(1, 10).resize(2, 5),
        Tensor[Int](1).fill(1)
      )

    val expectOutput = Tensor(T(5f, 10f))

    val output = Max(startFromZero = true).forward(input)
    output should be(expectOutput)
  }

  "Max keepDims" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    RandomGenerator.RNG.setSeed(10)
    val input =
      T(
        Tensor.range(1, 10).resize(2, 5),
        Tensor.scalar[Int](1)
      )

    val expectOutput = Tensor(T(5f, 10f)).resize(2, 1)

    val output = Max(true, true).forward(input)
    output should be(expectOutput)
  }

  "Max dim start from 1" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    RandomGenerator.RNG.setSeed(10)
    val input =
      T(
        Tensor.range(1, 10).resize(2, 5),
        Tensor.scalar[Int](2)
      )

    val expectOutput = Tensor(T(5f, 10f)).resize(2, 1)

    val output = Max(true, false).forward(input)
    output should be(expectOutput)
  }
}

class MaxSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val max = Max[Float, Float](startFromZero = true).setName("max_pool")
    val input1 = Tensor[Float].range(1, 6).resize(2, 3)
    val input2 = Tensor.scalar[Int](1)
    val input = T(input1, input2)
    runSerializationTest(max, input)
  }
}
