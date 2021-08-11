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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class PackSpec extends FlatSpec with Matchers {

  "Pack" should "compute correct output and gradient" in {
    val module1 = new Pack[Double](1)
    val module2 = new Pack[Double](2)
    val module3 = new Pack[Double](3)

    val input1 = Tensor[Double](2, 2)
    input1(Array(1, 1)) = 1
    input1(Array(1, 2)) = 2
    input1(Array(2, 1)) = 3
    input1(Array(2, 2)) = 4

    val input2 = Tensor[Double](2, 2)
    input2(Array(1, 1)) = 5
    input2(Array(1, 2)) = 6
    input2(Array(2, 1)) = 7
    input2(Array(2, 2)) = 8

    val input = T()
    input(1) = input1
    input(2) = input2

    val expectOutput1 = Tensor[Double](2, 2, 2)
    expectOutput1(Array(1, 1, 1)) = 1
    expectOutput1(Array(1, 1, 2)) = 2
    expectOutput1(Array(1, 2, 1)) = 3
    expectOutput1(Array(1, 2, 2)) = 4
    expectOutput1(Array(2, 1, 1)) = 5
    expectOutput1(Array(2, 1, 2)) = 6
    expectOutput1(Array(2, 2, 1)) = 7
    expectOutput1(Array(2, 2, 2)) = 8

    val expectOutput2 = Tensor[Double](2, 2, 2)
    expectOutput2(Array(1, 1, 1)) = 1
    expectOutput2(Array(1, 1, 2)) = 2
    expectOutput2(Array(1, 2, 1)) = 5
    expectOutput2(Array(1, 2, 2)) = 6
    expectOutput2(Array(2, 1, 1)) = 3
    expectOutput2(Array(2, 1, 2)) = 4
    expectOutput2(Array(2, 2, 1)) = 7
    expectOutput2(Array(2, 2, 2)) = 8

    val expectOutput3 = Tensor[Double](2, 2, 2)
    expectOutput3(Array(1, 1, 1)) = 1
    expectOutput3(Array(1, 1, 2)) = 5
    expectOutput3(Array(1, 2, 1)) = 2
    expectOutput3(Array(1, 2, 2)) = 6
    expectOutput3(Array(2, 1, 1)) = 3
    expectOutput3(Array(2, 1, 2)) = 7
    expectOutput3(Array(2, 2, 1)) = 4
    expectOutput3(Array(2, 2, 2)) = 8


    val output1 = module1.forward(input)
    val output2 = module2.forward(input)
    val output3 = module3.forward(input)

    output1 should be(expectOutput1)
    output2 should be(expectOutput2)
    output3 should be(expectOutput3)

    val gradInput1 = module1.backward(input, output1)
    val gradInput2 = module2.backward(input, output2)
    val gradInput3 = module3.backward(input, output3)

    gradInput1 should be(input)
    gradInput2 should be(input)
    gradInput3 should be(input)
  }

  "Pack" should "work with tensor input" in {
    val module1 = new Pack[Double](1)

    val input1 = Tensor[Double](2, 2)
    input1(Array(1, 1)) = 1
    input1(Array(1, 2)) = 2
    input1(Array(2, 1)) = 3
    input1(Array(2, 2)) = 4

    val output1 = module1.forward(input1)

    val expectOutput1 = Tensor[Double](1, 2, 2)
    expectOutput1(Array(1, 1, 1)) = 1
    expectOutput1(Array(1, 1, 2)) = 2
    expectOutput1(Array(1, 2, 1)) = 3
    expectOutput1(Array(1, 2, 2)) = 4

    val gradInput1 = module1.backward(input1, output1)

    output1 should be(expectOutput1)
    gradInput1 should be(input1)
  }
}

class PackSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val pack = new Pack[Float](1).setName("pack")
    val input1 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    val input = T()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(pack, input)
  }
}
