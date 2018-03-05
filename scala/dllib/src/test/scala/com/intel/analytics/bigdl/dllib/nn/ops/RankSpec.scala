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

class RankSpec extends FlatSpec with Matchers {
  "Rank Float operation" should "works correctly" in {
    val input =
        Tensor[Float](T(1f, 2f, 2f))

    val expectOutput = Tensor.scalar(1)

    val output = Rank[Float]().forward(input)
    output should be(expectOutput)
  }

  "Rank Boolean operation" should "works correctly" in {
    val input =
        Tensor[Boolean](T(true, true, false))

    val expectOutput = Tensor.scalar(1)

    val output = Rank[Float]().forward(input)
    output should be(expectOutput)
  }

  "Rank Double operation" should "works correctly" in {
    val input =
        Tensor[Double](T(2.0, 3.0, 2.0))

    val expectOutput = Tensor.scalar(1)

    val output = Rank[Float]().forward(input)
    output should be(expectOutput)
  }

  "Rank Char operation" should "works correctly" in {
    val input =
        Tensor[Char](T('b', 'c', 'a'))

    val expectOutput = Tensor.scalar(1)

    val output = Rank[Float]().forward(input)
    output should be(expectOutput)
  }

  "Rank Long operation" should "works correctly" in {
    val input =
        Tensor[Long](T(2L, 3L, 2L))

    val expectOutput = Tensor.scalar(1)

    val output = Rank[Float]().forward(input)
    output should be(expectOutput)
  }

  "Rank String operation" should "works correctly" in {
    val input =
        Tensor[String](T("aaa", "ccc", "aaa"))

    val expectOutput = Tensor.scalar(1)

    val output = Rank[Float]().forward(input)
    output should be(expectOutput)
  }

  "Rank Short operation" should "works correctly" in {
    val input =
        Tensor[Short](T(2: Short, 3: Short, 2: Short))

    val expectOutput = Tensor.scalar(1)

    val output = Rank[Float]().forward(input)
    output should be(expectOutput)
  }

  "Rank Int operation" should "works correctly" in {
    val input =
        Tensor[Int](T(2, 3, 2))

    val expectOutput = Tensor.scalar(1)

    val output = Rank[Float]().forward(input)
    output should be(expectOutput)
  }
}

class RankSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val rank = Rank[Float].setName("rank")
    val input = Tensor[Float](3, 3).apply1(_ => Random.nextFloat())
    runSerializationTest(rank, input, rank.
      asInstanceOf[ModuleToOperation[Float]].module.getClass)
  }
}
