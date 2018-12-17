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

class NegativeSpec extends FlatSpec with Matchers {
  "Negative forward" should "be correct" in {
    val input = Tensor[Double](T(1, 2, 3))
    val m = Negative[Float]()
    m.forward(input) should be(Tensor[Double](T(-1, -2, -3)))
  }

  "Negative backward" should "be correct" in {
    val input = Tensor[Double](T(1, 2, 3))
    val grad = Tensor[Double](T(2, 3, 4))
    val m = Negative[Float]()
    m.backward(input, grad) should be(Tensor[Double](T(-2, -3, -4)))
  }
}

class NegativeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val negative = Negative[Float]().setName("negative")
    val input = Tensor[Float](10).apply1(e => Random.nextFloat())
    runSerializationTest(negative, input)
  }
}
