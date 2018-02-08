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

import scala.util.Random

class ShapeSpec extends FlatSpec with Matchers {

  "Shape forward" should "be success" in {
    val layer = Shape()
    val input = Tensor(T(T(0.1f, 0.2f), T(0.1f, 0.2f), T(0.1f, 0.2f)))
    layer.forward(input) should be(Tensor[Int](T(3, 2)))
  }

  "Shape backward" should "be correct" in {
    val layer = Shape()
    val input = Tensor(T(T(0.1f, 0.2f), T(0.1f, 0.2f), T(0.1f, 0.2f)))
    val gradOutput = Tensor[Int](T(3, 2))
    layer.forward(input) should be(Tensor[Int](T(3, 2)))
    layer.backward(input, gradOutput) should be(Tensor(T(
      T(0.0f, 0.0f),
      T(0.0f, 0.0f),
      T(0.0f, 0.0f)
    )))
  }
}

class ShapeSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val shape = Shape[Float]().setName("shape")
    val input = Tensor[Float](3).apply1(_ => Random.nextFloat())
    runSerializationTest(shape, input)
  }
}
