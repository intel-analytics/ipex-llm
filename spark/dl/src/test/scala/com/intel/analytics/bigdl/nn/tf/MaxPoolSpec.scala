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

import com.intel.analytics.bigdl.nn.ops.ModuleToOperation
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

import scala.util.Random

class MaxPoolSpec extends FlatSpec with Matchers {
  "MaxPool operation VALID padding" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val expectOutput = Tensor(
      T(
        T(
          T(T(7.0, 8.0, 9.0)),
          T(T(7.0, 8.0, 9.0))
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


    val output = MaxPool[Double](
      Array(1, 2, 3, 1),
      Array(1, 2, 1, 1),
      "VALID").forward(input)

    output should equal(expectOutput)
  }

  "MaxPool operation SAME padding" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    val expectOutput = Tensor(
      T(
        T(
          T(
            T(4.0, 5.0, 6.0),
            T(7.0, 8.0, 9.0),
            T(7.0, 8.0, 9.0)
          ),
          T(
            T(4.0, 5.0, 6.0),
            T(7.0, 8.0, 9.0),
            T(7.0, 8.0, 9.0)
          )
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


    val output = MaxPool[Double](
      Array(1, 2, 3, 1),
      Array(1, 2, 1, 1),
      "SAME"
    ).forward(input)

    output should equal(expectOutput)
  }
}

class MaxPoolSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val maxPool = MaxPool[Float](
      Array(1, 2, 3, 1),
      Array(1, 2, 1, 1),
      "VALID").setName("maxPool")
    val input = Tensor[Float](1, 4, 3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(maxPool, input, maxPool.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
