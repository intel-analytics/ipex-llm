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

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class PowerSpec extends FlatSpec with Matchers {
  "A Power" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(1.0, 4, 9, 16, 25, 36)), 1, Array(2, 3))

    val power = new Power[Double](2)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A float Power" should "generate correct output" in {
    val input = Tensor(Storage[Float](Array(1.0f, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(1.0f, 4, 9, 16, 25, 36)), 1, Array(2, 3))

    val power = new Power[Float](2)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A Power with scale" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(4.0, 16, 36, 64, 100, 144)), 1, Array(2, 3))

    val power = new Power[Double](2, 2)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A Power with shift" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(0.0, 1, 2, 3, 4, 5)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(1.0, 4, 9, 16, 25, 36)), 1, Array(2, 3))

    val power = new Power[Double](2, 1, 1)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A Power with scale and shift" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(0.0, 1, 2, 3, 4, 5)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(1.0, 9, 25, 49, 81, 121)), 1, Array(2, 3))

    val power = new Power[Double](2, 2, 1)

    val powerOutput = power.forward(input)

    powerOutput should be (output)
  }

  "A Power" should "generate correct grad" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val gradOutput = Tensor(Storage(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)), 1, Array(2, 3))

    val power = new Power[Double](2, 2, 2)

    val output = power.forward(input)
    val gradInput = power.backward(input, gradOutput)

    output should be (Tensor(Storage(Array(16.0, 36, 64, 100, 144, 196)), 1, Array(2, 3)))
    gradInput should be (Tensor(Storage(Array(1.6, 4.8, 9.6, 16, 24, 33.6)), 1, Array(2, 3)))

  }

  "A Power" should "generate correct output and grad" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val gradOutput = Tensor(Storage(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)), 1, Array(2, 3))

    val power = new Power[Double](1, -1)

    val output = power.forward(input)
    val gradInput = power.backward(input, gradOutput)

    output should be (Tensor(Storage(Array(-1.0, -2, -3, -4, -5, -6)), 1, Array(2, 3)))
    gradInput should be (Tensor(Storage(Array(-0.1, -0.2, -0.3, -0.4, -0.5, -0.6)), 1, Array(2, 3)))

  }

  "A Power(3, 2, 2)" should "generate correct output and grad" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val gradOutput = Tensor(Storage(Array(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)), 1, Array(2, 3))

    val power = new Power[Double](3, 2, 2)

    val output = power.forward(input)
    val gradInput = power.backward(input, gradOutput)

    output should be (Tensor(Storage(Array(64.0, 216, 512, 1000, 1728, 2744)), 1, Array(2, 3)))
    gradInput should be (Tensor(Storage(Array(9.6, 43.2, 115.2, 240, 432, 705.6)), 1, Array(2, 3)))

  }

}

class PowerSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val power = Power[Float](2.0).setName("power")
    val input = Tensor[Float](2, 2).apply1(e => Random.nextFloat())
    runSerializationTest(power, input)
  }
}
