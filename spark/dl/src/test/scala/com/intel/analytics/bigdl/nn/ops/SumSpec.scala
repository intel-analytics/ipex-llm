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

class SumSpec extends FlatSpec with Matchers {
  "Sum operation" should "works correctly" in {
    val input =
      Tensor[Int](T(
        T(1, 1, 1),
        T(1, 1, 1)
      ))
    val op = Sum[Float, Int]()
    op.forward(T(input, Tensor[Int]())) should be(input)
    op.forward(T(input, Tensor.scalar[Int](1))) should be(Tensor[Int](T(2, 2, 2)))
    op.forward(T(input, Tensor[Int](T(1)))) should be(Tensor[Int](T(2, 2, 2)))
    op.forward(T(input, Tensor.scalar[Int](2))) should be(Tensor[Int](T(3, 3)))
    val op1 = Sum[Float, Int](keepDims = true)
    op1.forward(T(input, Tensor.scalar[Int](2))) should be(Tensor[Int](T(T(3), T(3))))
    op.forward(T(input, Tensor[Int](T(1, 2)))) should be(Tensor.scalar[Int](6))
  }
}

class SumSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val sum = Sum[Float, Float]().setName("sumOps")
    val input = T(Tensor[Float](2, 2).apply1(_ => Random.nextFloat()),
      Tensor[Float]())
    runSerializationTest(sum, input)
  }
}
