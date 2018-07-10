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
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class MMSpec extends FlatSpec with Matchers {
  "hashcode()" should "behave correctly" in {
    val m1 = new MM[Double]()
    val m2 = new MM[Double]()
    val m3 = new MM[Double](true, true)
    val m4 = new MM[Double]()
    val log = new Log[Double]()
    com.intel.analytics.bigdl.tensor.Tensor
    val input1 = Tensor[Double](3, 3).randn()
    val input2 = Tensor[Double](3, 3).randn()
    val input = T(1 -> input1, 2 -> input2)
    m4.forward(input)

    m1.hashCode() should equal(m2.hashCode())
    m1.hashCode() should not equal null
    m1.hashCode() should not equal log.hashCode()
    m1.hashCode() should not equal m3.hashCode()
    m1.hashCode() should not equal m4.hashCode()
  }

  "equals()" should "behave correctly" in {
    val m1 = new MM[Double]()
    val m2 = new MM[Double]()
    val m3 = new MM[Double](true, true)
    val m4 = new MM[Double]()
    val log = new Log[Double]()
    com.intel.analytics.bigdl.tensor.Tensor
    val input1 = Tensor[Double](3, 3).randn()
    val input2 = Tensor[Double](3, 3).randn()
    val input = T(1 -> input1, 2 -> input2)
    m4.forward(input)

    m1 should equal(m2)
    m1 should not equal null
    m1 should not equal log
    m1 should not equal m3
    m1 should not equal m4
  }

  "MM forward multi times" should "work properly" in {
    val mm = MM[Float]()
    val input1 = Tensor[Float](2, 3, 3).randn()
    val input2 = Tensor[Float](2, 3, 3).randn()
    val input = T(1 -> input1, 2 -> input2)

    val res1 = Tensor[Float](2, 3, 3)

    val res2 = Tensor[Float](2, 3, 3)

    res1.copy(mm.forward(input))

    res2.copy(mm.forward(input))

    res1 should be (res2)
  }

  "MM backward multi times" should "work properly" in {
    val mm = MM[Float]()
    val input1 = Tensor[Float](2, 3, 3).randn()
    val input2 = Tensor[Float](2, 3, 3).randn()
    val input = T(1 -> input1, 2 -> input2)

    val gradOutput = Tensor[Float](2, 3, 3).randn()

    val bres1 = mm.backward(input, gradOutput)

    val res1 = T(1 -> Tensor[Float](2, 3, 3).copy(bres1(1)),
      2 -> Tensor[Float](2, 3, 3).copy(bres1(2)))

    val bres2 = mm.backward(input, gradOutput)

    val res2 = T(1 -> Tensor[Float](2, 3, 3).copy(bres2(1)),
      2 -> Tensor[Float](2, 3, 3).copy(bres2(2)))

    res1 should be (res2)

  }
}

class MMSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val mm = MM[Float]().setName("mm_layer")
    val input1 = Tensor[Float](2, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](3, 4).apply1(e => Random.nextFloat())
    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(mm, input)
  }
}
