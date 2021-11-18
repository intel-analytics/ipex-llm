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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.Sequential._
import com.intel.analytics.bigdl.dllib.keras.serializer.ModuleSerializationTest

class EmbeddingSpec extends KerasBaseSpec {

  // Compared results with Keras on Python side
  "Embedding with weights" should "work properly" in {
    val weights = Tensor[Float](10, 32).rand()
    val seq = Sequential[Float]()
    val layer = Embedding[Float](10, 32, weights = weights, inputLength = 4)
    seq.add(layer)
    require(seq.getWeightsBias().sameElements(Array(weights)))
    seq.getOutputShape().toSingle().toArray should be (Array(-1, 4, 32))
    val input = Tensor[Float](2, 4)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 4
    input(Array(1, 4)) = 5
    input(Array(2, 1)) = 4
    input(Array(2, 2)) = 3
    input(Array(2, 3)) = 2
    input(Array(2, 4)) = 6
    val output = seq.forward(input).toTensor[Float]
    for (i <- 0 to 1) {
      val nonBatchOutput = output.split(1)(i)
      for (j <- 0 to 3) {
        val actual = nonBatchOutput.split(1)(j)
        val expected = weights.select(1, input.valueAt(i + 1, j + 1).toInt + 1)
        require(actual == expected)
      }
    }
    val gradInput = seq.backward(input, output)
  }

}

class EmbeddingSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = Embedding[Float](1000, 32, inputLength = 4)
    layer.build(Shape(2, 4))
    val input = Tensor[Float](2, 4)
    input(Array(1, 1)) = 1
    input(Array(1, 2)) = 2
    input(Array(1, 3)) = 4
    input(Array(1, 4)) = 5
    input(Array(2, 1)) = 4
    input(Array(2, 2)) = 3
    input(Array(2, 3)) = 2
    input(Array(2, 4)) = 6
    runSerializationTest(layer, input)
  }
}
