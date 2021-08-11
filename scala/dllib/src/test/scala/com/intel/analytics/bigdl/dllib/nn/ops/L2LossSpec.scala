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
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class L2LossSpec extends FlatSpec with Matchers {
  "L2Loss Double operation" should "works correctly" in {
    val input =
      Tensor[Double](
        T(
          T(1.5, 2.1, 2.9),
          T(0.5, 1.1, 1.9)
        ))

    val expectOutput = Tensor[Double](T(10.07))

    val output = L2Loss[Double]().forward(input)
    output should be(expectOutput)
  }

  "L2Loss Float operation" should "works correctly" in {
    val input =
      Tensor[Float](
        T(
          T(1.5f, 2.1f, 2.9f),
          T(0.5f, 1.1f, 1.9f)
        ))

    val expectOutput = Tensor[Float](T(10.07f))

    val output = L2Loss[Float]().forward(input)
    output should be(expectOutput)
  }
}

class L2LossSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val l2loss = L2Loss[Float]().setName("l2loss")
    val input = Tensor[Float](2, 5).apply1(_ => Random.nextFloat())
    runSerializationTest(l2loss, input,
      l2loss.asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
