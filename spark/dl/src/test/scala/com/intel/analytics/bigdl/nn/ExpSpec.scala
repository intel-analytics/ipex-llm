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
class ExpSpec extends FlatSpec with Matchers {
  "A Exp" should "generate correct output" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val output = Tensor(Storage(Array(
      2.718281828459045, 7.38905609893065, 20.085536923187668,
      54.598150033144236, 148.4131591025766, 403.4287934927351)), 1, Array(2, 3))

    val exp = new Exp[Double]()

    val powerOutput = exp.forward(input)

    powerOutput should equal (output)
  }

  "A Exp" should "generate correct gradInput" in {
    val input = Tensor(Storage[Double](Array(1.0, 2, 3, 4, 5, 6)), 1, Array(2, 3))

    val gradOutput = Tensor(Storage(Array(
      2.7183, 7.3891, 20.0855,
      54.5982, 148.4132, 403.4288)), 1, Array(2, 3))

    val exp = new Exp[Double]()

    exp.forward(input)
    val gradInput = exp.backward(input, gradOutput)
    val expectedGradInput = Tensor(Storage(Array(
      7.389105494300223, 54.59847442060847, 403.4280518706859,
      2980.9607151396153, 22026.47186452252, 162754.79404422196)), 1, Array(2, 3))

    gradInput should equal (expectedGradInput)
  }
}

class ExpSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val exp = Exp[Float]().setName("exp")
    val input = Tensor[Float](10).apply1(_ => Random.nextFloat())
    runSerializationTest(exp, input)
  }
}
