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
class CAddTableSpec extends FlatSpec with Matchers {
  "CAddTable" should "be correct when input is scalar" in {
    val module = CAddTable[Float]()
    val scalar = Tensor[Float](Array(2.0f), Array[Int]())
    val tensor = Tensor[Float](T(1, 2, 3))
    module.forward(T(scalar, tensor)) should be(Tensor[Float](T(3, 4, 5)))
    val grads = module.backward(T(scalar, tensor), Tensor[Float](T(1, 2, 3)))
    grads[Tensor[Float]](1).value() should be(6)
    grads[Tensor[Float]](2) should be(Tensor[Float](T(1, 2, 3)))
  }

  "CAddTable" should "be correct when input is scalar exchange order" in {
    val module = CAddTable[Float]()
    val scalar = Tensor[Float](Array(2.0f), Array[Int]())
    val tensor = Tensor[Float](T(1, 2, 3))
    module.forward(T(tensor, scalar)) should be(Tensor[Float](T(3, 4, 5)))
    val grads = module.backward(T(tensor, scalar), Tensor[Float](T(1, 2, 3)))
    grads[Tensor[Float]](1) should be(Tensor[Float](T(1, 2, 3)))
    grads[Tensor[Float]](2).value() should be(6)
  }

  "CAddTable with different size" should "be correct" in {
    val input1 = Tensor[Float](T(T(-0.52817175, -1.07296862, 0.86540763, -2.3015387,
      1.74481176, -0.7612069, 0.3190391, -0.24937038),
      T( 1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944, -1.09989127,
        -0.17242821, -0.87785842)))
    val input2 = Tensor[Float](T(T(1.62434536), T(-0.61175641)))
    val input3 = Tensor[Float](T(T(1.62434536, 1.62434536, 1.62434536, 1.62434536,
      1.62434536, 1.62434536, 1.62434536, 1.62434536),
      T(-0.61175641, -0.61175641, -0.61175641, -0.61175641, -0.61175641,
        -0.61175641, -0.61175641, -0.61175641)))
    val layer = CAddTable[Float]()
    val output = layer.forward(T(input1, input2))
    val output2 = layer.forward(T(input1, input3))
    output should be(output2)

    val gradInput = layer.backward(T(input1, input2), output)
    val gradInput2 = layer.backward(T(input1, input3), output2)

    gradInput[Tensor[Float]](1) should be(gradInput2[Tensor[Float]](1))
    gradInput[Tensor[Float]](2) should be(gradInput2[Tensor[Float]](2))
  }
}

class CAddTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    val caddTable = CAddTable[Float](false).setName("caddTable")
    runSerializationTest(caddTable, input)
  }
}
