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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class Conv2DSpec extends FlatSpec with Matchers {
  "Add operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val expectOutput = Tensor(
      T(
        T(
          T(
            T(138.0, 216.0),
            T( 288.0, 459.0),
            T( 246.0, 396.0)),
          T(
            T( 138.0, 216.0),
            T( 288.0, 459.0),
            T( 246.0, 396.0))
        )
      ))

    val input =
      Tensor(
        T(
          T(
            T(
              T(1.0, 2.0, 3.0),
              T(4.0, 5.0, 6.0),
              T(7.0, 8.0, 9.0)),
            T(
              T(1.0, 2.0, 3.0),
              T(4.0, 5.0, 6.0),
              T(7.0, 8.0, 9.0)),
            T(
              T(1.0, 2.0, 3.0),
              T(4.0, 5.0, 6.0),
              T(7.0, 8.0, 9.0)),
            T(
              T(1.0, 2.0, 3.0),
              T(4.0, 5.0, 6.0),
              T(7.0, 8.0, 9.0))
          )
        )
      )

    val filter = Tensor(
      T(
        T(
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)),
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)),
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0))),
        T(
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)),
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)),
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0))),
        T(
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)),
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)),
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0))),
        T(
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)),
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)),
          T(T(1.0, 3.0), T(2.0, 3.0), T(3.0, 4.0)))
      ))


    val output = Conv2D[Double](2, 1, -1, -1).forward(T(input, filter))
    output should equal(expectOutput)
  }
}

class Conv2DSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val conv2d = Conv2D[Float](2, 1, -1, -1).setName("conv2d")
    val inputTensor = Tensor[Float](1, 4, 3, 3).apply1(_ => Random.nextFloat())
    val filter = Tensor[Float](4, 3, 3, 2).apply1(_ => Random.nextFloat())
    val input = T(inputTensor, filter)
    runSerializationTest(conv2d, input)
  }
}
