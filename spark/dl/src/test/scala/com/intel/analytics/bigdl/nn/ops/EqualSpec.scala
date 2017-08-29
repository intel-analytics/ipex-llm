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
import org.scalatest.{FlatSpec, Matchers}

class EqualSpec extends FlatSpec with Matchers {
  "Equal Float operation" should "works correctly" in {
    val input =
      T(
        Tensor[Float](T(1f, 2f, 2f)),
        Tensor[Float](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Equal[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "Equal Boolean operation" should "works correctly" in {
    val input =
      T(
        Tensor[Boolean](T(true, true, false)),
        Tensor[Boolean](T(false, true, false))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Equal[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "Equal Double operation" should "works correctly" in {
    val input =
      T(
        Tensor[Double](T(1f, 2f, 2f)),
        Tensor[Double](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Double](T(false, false, true))

    val output = Equal[Double]().forward(input)
    output should be(expectOutput)
  }

  "Equal Char operation" should "works correctly" in {
    val input =
      T(
        Tensor[Char](T(1f, 2f, 2f)),
        Tensor[Char](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Equal[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "Equal Long operation" should "works correctly" in {
    val input =
      T(
        Tensor[Long](T(1f, 2f, 2f)),
        Tensor[Long](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Equal[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "Equal String operation" should "works correctly" in {
    val input =
      T(
        Tensor[String](T(1f, 2f, 2f)),
        Tensor[String](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Equal[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "Equal Short operation" should "works correctly" in {
    val input =
      T(
        Tensor[Short](T(1f, 2f, 2f)),
        Tensor[Short](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Equal[Boolean]().forward(input)
    output should be(expectOutput)
  }

  "Equal Int operation" should "works correctly" in {
    val input =
      T(
        Tensor[Int](T(1f, 2f, 2f)),
        Tensor[Int](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Equal[Boolean]().forward(input)
    output should be(expectOutput)
  }
}
