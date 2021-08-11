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

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest

class GatherSpec extends FlatSpec with Matchers {
  "gather with scalar" should "works fine" in {
    val gather = Gather[Float, Float]()
    val indices = Tensor[Int](Array(1), Array[Int]())
    val input = Tensor.range(1, 24).resize(2, 3, 4)
    val output = gather.forward(T(input, indices))

    output should be (input.narrow(1, 2, 1))
  }

  "gather with 1-D tensor" should "works fine" in {
    val gather = Gather[Float, Float]()
    val indices = Tensor[Int](T(0, 1))
    val input = Tensor.range(1, 24).resize(2, 3, 4)
    val output = gather.forward(T(input, indices))

    output should be (input)
  }

  "gather with 2-D tensor" should "works fine" in {
    val gather = Gather[Float, Float]()
    val indices = Tensor[Int](T(T(0, 1), T(0, 1)))
    val input = Tensor.range(1, 24).resize(2, 3, 4)
    val output = gather.forward(T(input, indices))

    val exceptedOutput = Tensor(2, 2, 3, 4)
    exceptedOutput.select(1, 1).copy(input)
    exceptedOutput.select(1, 2).copy(input)

    output should be (exceptedOutput)
  }

  "gather with Float input" should "works fine" in {
    val gather = Gather[Float, Float]()
    val indices = Tensor[Float](T(T(0f, 1f), T(0f, 1f)))
    val input = Tensor.range(1, 24).resize(2, 3, 4)
    val output = gather.forward(T(input, indices))

    val exceptedOutput = Tensor(2, 2, 3, 4)
    exceptedOutput.select(1, 1).copy(input)
    exceptedOutput.select(1, 2).copy(input)

    output should be (exceptedOutput)
  }

  "gather with scalar and specific dimension" should "works fine" in {
    val gather = Gather[Float, Float](2)
    val indices = Tensor[Int](Array(1), Array[Int]())
    val input = Tensor.range(1, 24).resize(2, 3, 4)
    val output = gather.forward(T(input, indices))

    output should be (input.narrow(2, 2, 1))
  }

  "gather with scalar from last dimension" should "works fine" in {
    val gather = Gather[Float, Float](0)
    val indices = Tensor[Int](Array(1), Array[Int]())
    val input = Tensor.range(1, 24).resize(2, 3, 4, 5)
    val output = gather.forward(T(input, indices))

    output should be (input.narrow(4, 2, 1))
  }

  "gather with scalar from negative dimension" should "works fine" in {
    val gather = Gather[Float, Float](-2)
    val indices = Tensor[Int](Array(1), Array[Int]())
    val input = Tensor.range(1, 24).resize(2, 3, 4)
    val output = gather.forward(T(input, indices))

    output should be (input.narrow(1, 2, 1))
  }
}

class GatherSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val gather = Gather[Float, Float]().setName("gather")
    val input1 = Tensor[Float].range(1, 6).resize(2, 3)
    val input2 = Tensor[Int](2).fill(1)
    val input = T(input1, input2)
    runSerializationTest(gather, input)
  }
}
