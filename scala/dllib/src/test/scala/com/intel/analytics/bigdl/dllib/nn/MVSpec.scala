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
class MVSpec extends FlatSpec with Matchers {
  "hashcode()" should "behave correctly" in {
    val m1 = new MV[Double]()
    val m2 = new MV[Double]()
    val m3 = new MV[Double](true)
    val m4 = new MV[Double]()
    val log = new Log[Double]()
    com.intel.analytics.bigdl.tensor.Tensor
    val input1 = Tensor[Double](3, 3).randn()
    val input2 = Tensor[Double](3).randn()
    val input = T(1 -> input1, 2 -> input2)
    m4.forward(input)

    m1.hashCode() should equal(m2.hashCode())
    m1.hashCode() should not equal null
    m1.hashCode() should not equal log.hashCode()
    m1.hashCode() should not equal m3.hashCode()
    m1.hashCode() should not equal m4.hashCode()

  }

  "equals()" should "behave correctly" in {
    val m1 = new MV[Double]()
    val m2 = new MV[Double]()
    val m3 = new MV[Double](true)
    val m4 = new MV[Double]()
    val log = new Log[Double]()
    com.intel.analytics.bigdl.tensor.Tensor
    val input1 = Tensor[Double](3, 3).randn()
    val input2 = Tensor[Double](3).randn()
    val input = T(1 -> input1, 2 -> input2)
    m4.forward(input)

    m1 should equal(m2)
    m1 should not equal null
    m1 should not equal log
    m1 should not equal m3
    m1 should not equal m4
  }
}

class MVSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val mv = MV[Float]().setName("mv_layer")
    val input1 = Tensor[Float](2, 3).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](3).apply1(e => Random.nextFloat())
    val input = new Table()
    input(1.0f) = input1
    input(2.0f) = input2
    runSerializationTest(mv, input)
  }
}
