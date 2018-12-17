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

class GreaterSpec extends FlatSpec with Matchers {
  "Greater Float operation" should "works correctly" in {
    val input =
      T(
        Tensor[Float](T(1f, 4f, 2f)),
        Tensor[Float](T(2f, 3f, 2f))
      )

    val expectOutput = Tensor[Boolean](T(false, true, false))

    val output = Greater[Float]().forward(input)
    output should be(expectOutput)
  }

  "Greater Double operation" should "works correctly" in {
    val input =
      T(
        Tensor[Double](T(1.0, 4.0, 2.0)),
        Tensor[Double](T(2.0, 3.0, 2.0))
      )

    val expectOutput = Tensor[Boolean](T(false, true, false))

    val output = Greater[Float]().forward(input)
    output should be(expectOutput)
  }

  "Greater Char operation" should "works correctly" in {
    val input =
      T(
        Tensor[Char](T('a', 'b', 'b')),
        Tensor[Char](T('b', 'c', 'a'))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Greater[Float]().forward(input)
    output should be(expectOutput)
  }

  "Greater Long operation" should "works correctly" in {
    val input =
      T(
        Tensor[Long](T(1L, 2L, 3L)),
        Tensor[Long](T(2L, 3L, 2L))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Greater[Float]().forward(input)
    output should be(expectOutput)
  }

  "Greater Short operation" should "works correctly" in {
    val input =
      T(
        Tensor[Short](T(1: Short, 2: Short, 3: Short)),
        Tensor[Short](T(2: Short, 3: Short, 2: Short))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Greater[Float]().forward(input)
    output should be(expectOutput)
  }

  "Greater Int operation" should "works correctly" in {
    val input =
      T(
        Tensor[Int](T(1, 2, 3)),
        Tensor[Int](T(2, 3, 2))
      )

    val expectOutput = Tensor[Boolean](T(false, false, true))

    val output = Greater[Float]().forward(input)
    output should be(expectOutput)
  }
}

class GreaterSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val greater = Greater[Float]().setName("greater")
    val input1 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input2 = Tensor[Float](5).apply1(_ => Random.nextFloat())
    val input = T(input1, input2)
    runSerializationTest(greater, input, greater.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
