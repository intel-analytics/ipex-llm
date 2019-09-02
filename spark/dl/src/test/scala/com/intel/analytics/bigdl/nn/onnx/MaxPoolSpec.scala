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

package com.intel.analytics.bigdl.nn.onnx

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{FlatSpec, Matchers}

class MaxPoolSpec extends FlatSpec with Matchers {
  "MaxPool" should "work" in {
    val kernelShape = List(2, 2)
    val maxPool = MaxPool[Float](kernelShape = kernelShape)
    val input = Tensor[Float](1, 3, 3)

    var counter = 1
    for (i <- 1 to 1) {
      for (j <- 1 to 3) {
        for (k <- 1 to 3) {
          input.setValue(i, j, k, counter)
          counter += 1
        }
      }
    }

    val expectedOut = Tensor[Float](1, 2, 2)
    expectedOut.setValue(1, 1, 1, 5)
    expectedOut.setValue(1, 1, 2, 6)
    expectedOut.setValue(1, 2, 1, 8)
    expectedOut.setValue(1, 2, 2, 9)

    val out = maxPool.forward(input)

    expectedOut should be(out)
  }
}


class MaxPoolSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val maxPool = AveragePool[Float](kernelShape = List(2, 2))
    val input = Tensor[Float](5, 5).rand()
    runSerializationTest(maxPool, input)
  }
}

