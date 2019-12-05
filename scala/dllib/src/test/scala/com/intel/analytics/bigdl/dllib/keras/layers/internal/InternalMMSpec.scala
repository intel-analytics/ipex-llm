/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers.internal

import com.intel.analytics.bigdl.nn.{Input, Log}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Shape, T}
import com.intel.analytics.zoo.pipeline.api.autograd.{AutoGrad, Variable}
import com.intel.analytics.zoo.pipeline.api.keras.models.Model
import com.intel.analytics.zoo.pipeline.api.keras.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}


class InternalMMSpec extends FlatSpec with Matchers {

  "mm forward multiple times" should "be ok" in {
    val input1 = Variable[Float](inputShape = Shape(4, 2))
    val input2 = Variable[Float](inputShape = Shape(4, 2))
    val result = AutoGrad.mm(input1, input2, axes = List(2, 2))
    val model = Model[Float](input = Array(input1, input2), output = result)
    val recordNum = 2
    val i1 = Tensor[Float](recordNum, 3, 4).rand()
    val i2 = Tensor[Float](recordNum, 3, 4).rand()
    val o1 = model.forward(T(i1, i2)).toTensor[Float].clone()
    val o2 = model.forward(T(i1, i2)).toTensor[Float].clone()
    assert(o1.almostEqual(o2, 1e-5))
  }

  "hashcode()" should "behave correctly" in {

    val input1 = Input[Float]()
    val input2 = Input[Float]()
    val model = InternalMM[Float](transA = true)
    val recordNum = 2
    val i1 = Tensor[Float](recordNum, 3, 4).rand()
    val i2 = Tensor[Float](recordNum, 3, 4).rand()
    val o1 = model.forward(T(i1, i2)).toTensor[Float].clone()
    val o2 = model.forward(T(i1, i2)).toTensor[Float].clone()
    assert(o1.almostEqual(o2, 1e-5))

  }

  "equals()" should "behave correctly" in {
    val m1 = new InternalMM[Double]()
    val m2 = new InternalMM[Double]()
    val m3 = new InternalMM[Double](true, true)
    val m4 = new InternalMM[Double]()
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
}

class InternalMMSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val mm = InternalMM[Float]().setName("mm_layer")
    val input1 = Tensor[Float](2, 3).rand()
    val input2 = Tensor[Float](3, 4).rand()
    runSerializationTest(mm, T(input1, input2))
  }
}
