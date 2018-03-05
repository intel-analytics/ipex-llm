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

class IndicatorColSpec extends FlatSpec with Matchers {

  "IndicatorColSpec Operation with isCount=true" should "work correctly" in {

    val input = Tensor.sparse(
      Array(Array(0, 1, 1, 2, 2, 3, 3, 3),
        Array(0, 0, 3, 0, 1, 0, 1, 2)),
      Array(3, 1, 2, 0, 3, 1, 2, 2),
      Array(4, 4)
    )

    val expectedOutput = Tensor[Double](
      T(T(0, 0, 0, 1),
        T(0, 1, 1, 0),
        T(1, 0, 0, 1),
        T(0, 1, 2, 0)))

    val output = IndicatorCol[Double](
      feaLen = 4, isCount = true
    ).forward(input)

    output should be(expectedOutput)
  }

  "IndicatorColSpec Operation with isCount=false" should "work correctly" in {

    val input = Tensor.sparse(
      Array(Array(0, 1, 1, 2, 2, 3, 3, 3),
        Array(0, 0, 3, 0, 1, 0, 1, 2)),
      Array(3, 1, 2, 0, 3, 1, 2, 2),
      Array(4, 4)
    )

    val expectedOutput = Tensor[Float](
      T(T(0, 0, 0, 1),
        T(0, 1, 1, 0),
        T(1, 0, 0, 1),
        T(0, 1, 1, 0)))

    val output = IndicatorCol[Float](
      feaLen = 4, isCount = false
    ).forward(input)

    output should be(expectedOutput)
  }
}

class IndicatorColSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val indicatorCol = IndicatorCol[Float](
      feaLen = 4,
      isCount = true
    ).setName("indicatorCol")
    val input = Tensor.sparse(
      Array(Array(0, 1, 1, 2, 2, 3, 3, 3),
        Array(0, 0, 3, 0, 1, 0, 1, 2)),
      Array(3, 1, 2, 0, 3, 1, 2, 2),
      Array(4, 4)
    )
    runSerializationTest(indicatorCol, input)
  }
}
