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

class NegativeEntropyPenaltySpec extends FlatSpec with Matchers {

  "NegativeEntropyPenalty forward" should "be correct" in {
    val input = Tensor[Float](T(0.5, 0.2, 0.3))
    val m = NegativeEntropyPenalty[Float]()
    m.forward(input) should be(Tensor[Float](T(0.5, 0.2, 0.3)))
  }

  "NegativeEntropyPenalty backward" should "be correct" in {
    val input = Tensor[Float](T(0.5, 0.2, 0.3))
    val grad = Tensor[Float](T(0.4, 0.2, 0.3))
    val m = NegativeEntropyPenalty[Float]()
    val gradInput = m.backward(input, grad)
    def gradient(x: Double): Double = 0.01 * (math.log(x) + 1)
    val expected = Tensor[Float](T(0.4 + gradient(0.5),
                                   0.2 + gradient(0.2),
                                   0.3 + gradient(0.3)))
    gradInput.almostEqual(expected, 1e-5) should be (true)
  }
}

class NegativeEntropyPenaltySerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val penalty = NegativeEntropyPenalty[Float](0.01).setName("NegativeEntropyPenalty")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(penalty, input)
  }
}
