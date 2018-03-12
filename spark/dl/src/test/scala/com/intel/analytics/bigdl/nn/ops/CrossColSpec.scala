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

class CrossColSpec extends FlatSpec with Matchers {
  "CrossCol Operation with two feature columns" should "work correctly" in {
    val input = T(
      Tensor[String](T("A,D", "B", "A,C")),
      Tensor[String](T("1", "2", "3,4"))
    )

    val expectedOutput = Tensor.sparse(
      Array(Array(0, 0, 1, 2, 2, 2, 2),
        Array(0, 1, 0, 0, 1, 2, 3)),
      Array(80, 98, 50, 99, 27, 89, 33),
      Array(3, 4)
    )

    val output = CrossCol[Double](hashBucketSize = 100)
      .forward(input)

    output should be(expectedOutput)
  }
  "CrossCol Operation with more than two feature columns" should "work correctly" in {
    val input = T(
      Tensor[String](T("A,D", "B", "A,C")),
      Tensor[String](T("1", "2", "3,4")),
      Tensor[String](T("1", "2", "3"))
    )

    val expectedOutput = Tensor.sparse(
      Array(Array(0, 0, 1, 2, 2, 2, 2),
        Array(0, 1, 0, 0, 1, 2, 3)),
      Array(94, 34, 68, 82, 83, 97, 12),
      Array(3, 4)
    )

    val output = CrossCol[Double](hashBucketSize = 100)
      .forward(input)

    output should be(expectedOutput)
  }
}

class CrossColSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val crosscol = CrossCol[Float](hashBucketSize = 100)
      .setName("CrossCol")
    val input = T(
      Tensor[String](T("A,D", "B", "A,C")),
      Tensor[String](T("1", "2", "3,4"))
    )
    runSerializationTest(crosscol, input)
  }
}
