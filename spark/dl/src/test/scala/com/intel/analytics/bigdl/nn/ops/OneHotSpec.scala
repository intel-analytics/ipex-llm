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

class OneHotSpec extends FlatSpec with Matchers {
  "OneHot operation one dimension index" should "works correctly" in {
    val input =
      T(Tensor[Long](T(0, 2, -1, 1)),
        Tensor[Int](Array(3), shape = Array[Int]()),
        Tensor[Double](Array(0.5), shape = Array[Int]()),
        Tensor[Double](Array(0.0), shape = Array[Int]()))

    val expectOutput =
      Tensor[Double](T(
        T(0.5, 0.0, 0.0),
        T(0.0, 0.0, 0.5),
        T(0.0, 0.0, 0.0),
        T(0.0, 0.5, 0.0)
      ))

    val output = OneHot[Double, Double](
      axis = -1
    ).forward(input)

    output should be(expectOutput)
  }

  "OneHot operation two dimension index" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val input =
      T(
        Tensor[Long](T(T(0, 2), T(1, -1))),
        Tensor[Int](Array(3), shape = Array[Int]()),
        Tensor[Double](Array(1.0), shape = Array[Int]()),
        Tensor[Double](Array(0.0), shape = Array[Int]())
      )

    val expectOutput =
      Tensor(T(
        T(T(1.0, 0.0, 0.0),
        T(0.0, 0.0, 1.0)),
        T(T(0.0, 1.0, 0.0),
        T(0.0, 0.0, 0.0))
      ))

    val output = OneHot[Double, Double](
      axis = -1
    ).forward(input)

    output should be(expectOutput)
  }

  "OneHot operation two dimension index with type _" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val input =
      T(
        Tensor[Long](T(T(0, 2), T(1, -1))),
        Tensor[Int](Array(3), shape = Array[Int]()),
        Tensor[Double](Array(1.0), shape = Array[Int]()),
        Tensor[Double](Array(0.0), shape = Array[Int]())
      )

    val expectOutput =
      Tensor(T(
        T(T(1.0, 0.0, 0.0),
          T(0.0, 0.0, 1.0)),
        T(T(0.0, 1.0, 0.0),
          T(0.0, 0.0, 0.0))
      ))

    val output = OneHot[Double, Double](
      axis = -1
    ).forward(input)

    output should be(expectOutput)
  }
}

class OneHotSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val oneHot = OneHot[Float, Float](axis = -1).setName("oneHot")
    val input =
      T(Tensor[Long](T(0, 2, -1, 1)),
        Tensor[Int](Array(3), shape = Array[Int]()),
        Tensor[Float](Array(0.5f), shape = Array[Int]()),
        Tensor[Float](Array(0.0f), shape = Array[Int]()))
    runSerializationTest(oneHot, input, oneHot
      .asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
