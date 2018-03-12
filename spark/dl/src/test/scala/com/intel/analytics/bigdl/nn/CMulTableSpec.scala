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

class CMulTableSpec extends FlatSpec with Matchers {
  "CMulTable" should "be correct when input is scalar" in {
    val module = CMulTable[Float]()
    val scalar = Tensor[Float](Array(2.0f), Array[Int]())
    val tensor = Tensor[Float](T(1, 2, 3))
    module.forward(T(scalar, tensor)) should be(Tensor[Float](T(2, 4, 6)))
    val grads = module.backward(T(scalar, tensor), Tensor[Float](T(1, 2, 3)))
    grads[Tensor[Float]](1).value() should be(14)
    grads[Tensor[Float]](2) should be(Tensor[Float](T(2, 4, 6)))
  }

  "CMulTable" should "be correct when input is scalar exchange order" in {
    val module = CMulTable[Float]()
    val scalar = Tensor[Float](Array(2.0f), Array[Int]())
    val tensor = Tensor[Float](T(1, 2, 3))
    module.forward(T(tensor, scalar)) should be(Tensor[Float](T(2, 4, 6)))
    val grads = module.backward(T(tensor, scalar), Tensor[Float](T(1, 2, 3)))
    grads[Tensor[Float]](1) should be(Tensor[Float](T(2, 4, 6)))
    grads[Tensor[Float]](2).value() should be(14)
  }
}

class CMulTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2

    val cmulTable = CMulTable[Float]().setName("cmulTable")
    runSerializationTest(cmulTable, input)
  }
}
