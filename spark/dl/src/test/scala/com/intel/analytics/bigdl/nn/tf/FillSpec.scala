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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class FillSpec extends FlatSpec with Matchers {

  "Fill forward" should "be correct" in {
    val layer = Fill()
    val shape = Tensor[Int](T(2, 3))
    val value = Tensor[Float](Array(0.1f), Array[Int]())
    layer.forward(T(shape, value)) should be(Tensor(T(T(0.1f, 0.1f, 0.1f), T(0.1f, 0.1f, 0.1f))))
  }

  "Fill forward scalar" should "be correct" in {
    val layer = Fill[Double]()
    val shape = Tensor[Int]()
    val value = Tensor[Float](Array(0.1f), Array[Int]())
    layer.forward(T(shape, value)) should be(Tensor.scalar[Float](0.1f))
  }

  "Fill backward" should "be correct" in {
    val layer = Fill()
    val shape = Tensor[Int](T(2, 3))
    val value = Tensor[Float](Array(0.1f), Array[Int]())
    val gradOutput = Tensor(2, 3).rand()
    layer.forward(T(shape, value)) should be(Tensor(T(T(0.1f, 0.1f, 0.1f), T(0.1f, 0.1f, 0.1f))))
    val gradInput = layer.backward(T(shape, value), gradOutput)
    gradInput[Tensor[Int]](1) should be (Tensor[Int](2))
    gradInput[Tensor[Float]](2) should be (Tensor[Float](Array(0.0f), Array[Int]()))
  }
}

class FillSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val fill = Fill[Float]().setName("fill")
    val shape = Tensor[Int](T(2, 3))
    val value = Tensor[Float](Array(0.1f), Array[Int]())
    val input = T(shape, value)
    runSerializationTest(fill, input)
  }
}
