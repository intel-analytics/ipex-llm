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
package com.intel.analytics.bigdl.nn

import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.{T, Table}
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

import scala.util.Random

@com.intel.analytics.bigdl.tags.Parallel
class FlattenTableSpec extends FlatSpec with BeforeAndAfter with Matchers {
  "An FlattenTable" should "generate correct output and grad" in {
    val layer = new FlattenTable[Double]()
    val input = T(Tensor[Double](
      Storage(Array(1.0, 2, 3))),
      T(
        Tensor[Double](Storage(Array(4.0, 3, 2, 1))),
        T(
          Tensor[Double](Storage(Array(3.0, 2, 1)))
        )
      )
    )
    val expectedOutput = T(Tensor[Double](
      Storage(Array(1.0, 2, 3))), Tensor[Double](Storage(Array(4.0, 3, 2, 1))),
      Tensor[Double](Storage(Array(3.0, 2, 1)))
    )


    val expectedGradInput = T(
      Tensor[Double](Storage(Array(1.1, 2.0, 3))),
      T(
        Tensor[Double](Storage(Array(4.1, 3, 2, 1))),
        T(
          Tensor[Double](Storage(Array(3.1, 2, 1)))
        )
      )
    )
    val gradOutput = T(Tensor[Double](
      Storage(Array(1.1, 2, 3))), Tensor[Double](Storage(Array(4.1, 3, 2, 1))),
      Tensor[Double](Storage(Array(3.1, 2, 1)))
    )

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()

    output should be (expectedOutput)
    gradInput should be (expectedGradInput)
  }

  "An FlattenTable" should "also generate correct output and grad" in {
    val layer = new FlattenTable[Double]()
    val input = T(
      Tensor[Double](Storage(Array(1.0, 2, 3))),
      T(
        Tensor[Double](Storage(Array(4.0, 3, 2, 1))),
        Tensor[Double](Storage(Array(3.0, 2, 1))),
        T(
          Tensor[Double](Storage(Array(3.0, 2, 1))),
          Tensor[Double](Storage(Array(3.0, 2, 1)))
        ),
        T(
          Tensor[Double](Storage(Array(3.0, 2, 1))),
          Tensor[Double](Storage(Array(3.0, 2, 1)))
          )
      )
    )
    val expectedOutput = T(
      Tensor[Double](Storage(Array(1.0, 2, 3))),
      Tensor[Double](Storage(Array(4.0, 3, 2, 1))),
      Tensor[Double](Storage(Array(3.0, 2, 1))),
      Tensor[Double](Storage(Array(3.0, 2, 1))),
      Tensor[Double](Storage(Array(3.0, 2, 1))),
      Tensor[Double](Storage(Array(3.0, 2, 1))),
      Tensor[Double](Storage(Array(3.0, 2, 1)))
    )


    val expectedGradInput = T(
      Tensor[Double](Storage(Array(1.1, 2, 3))),
      T(
        Tensor[Double](Storage(Array(4.1, 3, 2, 1))),
        Tensor[Double](Storage(Array(3.1, 2, 1))),
        T(
          Tensor[Double](Storage(Array(3.1, 2, 1))),
          Tensor[Double](Storage(Array(3.1, 2, 1)))
        ),
        T(
          Tensor[Double](Storage(Array(3.1, 2, 1))),
          Tensor[Double](Storage(Array(3.1, 2, 1)))
        )
      )
    )
    val gradOutput = T(
      Tensor[Double](Storage(Array(1.1, 2, 3))),
      Tensor[Double](Storage(Array(4.1, 3, 2, 1))),
      Tensor[Double](Storage(Array(3.1, 2, 1))),
      Tensor[Double](Storage(Array(3.1, 2, 1))),
      Tensor[Double](Storage(Array(3.1, 2, 1))),
      Tensor[Double](Storage(Array(3.1, 2, 1))),
      Tensor[Double](Storage(Array(3.1, 2, 1)))
    )

    val start = System.nanoTime()
    val output = layer.forward(input)
    val gradInput = layer.backward(input, gradOutput)
    val end = System.nanoTime()

    output should be (expectedOutput)
    gradInput should be (expectedGradInput)
  }
}

class FlattenTableSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val flattenTable = FlattenTable[Float]().setName("flattenTable")
    val input1 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    val input2 = Tensor[Float](5, 5).apply1(e => Random.nextFloat())
    var input = new Table()
    input(1.toFloat) = input1
    input(2.toFloat) = input2
    runSerializationTest(flattenTable, input)
  }
}
