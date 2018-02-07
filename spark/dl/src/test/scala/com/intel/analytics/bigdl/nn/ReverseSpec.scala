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
import com.intel.analytics.bigdl.utils.RandomGenerator
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.collection.mutable
import scala.util.Random

@com.intel.analytics.bigdl.tags.Serial
class ReverseSpec extends FlatSpec with Matchers {

  "A Reverse()" should "generate correct output and grad for Tensor input dim1 inplace" in {
    def randomn(): Double = RandomGenerator.RNG.uniform(-10, 10)
    val layer = new Reverse[Double](1)

    val input = Tensor[Double](4, 3)
    input.apply1(x => randomn())
    val expectedOutput = Tensor[Double]().resizeAs(input)
    expectedOutput.select(1, 1).copy(input(4))
    expectedOutput.select(1, 2).copy(input(3))
    expectedOutput.select(1, 3).copy(input(2))
    expectedOutput.select(1, 4).copy(input(1))

    val gradOutput = Tensor[Double](4, 3)
    gradOutput.apply1(x => randomn())
    val expectedGradInput = Tensor[Double]().resizeAs(gradOutput)
    expectedGradInput(1).copy(gradOutput(4))
    expectedGradInput(2).copy(gradOutput(3))
    expectedGradInput(3).copy(gradOutput(2))
    expectedGradInput(4).copy(gradOutput(1))

    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)

    output should be (expectedOutput)
    gradInput should be (expectedGradInput)

  }

  "A Reverse()" should "generate correct output and grad for Tensor input dim1" in {
    def randomn(): Double = RandomGenerator.RNG.uniform(-10, 10)
    val layer = new Reverse[Double](1)

    val input = Tensor[Double](3, 3, 3)
    input.apply1(x => randomn())
    val expectedOutput = Tensor[Double]().resizeAs(input)
    expectedOutput(1).copy(input(3))
    expectedOutput(2).copy(input(2))
    expectedOutput(3).copy(input(1))

    val gradOutput = Tensor[Double](3, 3, 3)
    gradOutput.apply1(x => randomn())
    val expectedGradInput = Tensor[Double]().resizeAs(gradOutput)
    expectedGradInput(1).copy(gradOutput(3))
    expectedGradInput(2).copy(gradOutput(2))
    expectedGradInput(3).copy(gradOutput(1))

    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)

    output should be (expectedOutput)
    gradInput should be (expectedGradInput)

  }
  "A Reverse()" should "generate correct output and grad for Tensor input dim2" in {
    def randomn(): Double = RandomGenerator.RNG.uniform(-10, 10)
    val layer = new Reverse[Double](2)

    val input = Tensor[Double](3, 3, 3)
    input.apply1(x => randomn())
    val expectedOutput = Tensor[Double]().resizeAs(input)
    expectedOutput.select(2, 1).copy(input.select(2, 3))
    expectedOutput.select(2, 2).copy(input.select(2, 2))
    expectedOutput.select(2, 3).copy(input.select(2, 1))

    val gradOutput = Tensor[Double](3, 3, 3)
    gradOutput.apply1(x => randomn())
    val expectedGradInput = Tensor[Double]().resizeAs(gradOutput)
    expectedGradInput.select(2, 1).copy(gradOutput.select(2, 3))
    expectedGradInput.select(2, 2).copy(gradOutput.select(2, 2))
    expectedGradInput.select(2, 3).copy(gradOutput.select(2, 1))

    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)

    output should be (expectedOutput)
    gradInput should be (expectedGradInput)

  }
}

class ReverseSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val reverse = Reverse[Float]().setName("reverse")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(reverse, input)
  }
}
