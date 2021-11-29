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

import com.intel.analytics.bigdl.dllib.nn.LookupTableSparse
import com.intel.analytics.bigdl.dllib.nn.abstractnn.AbstractModule
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{Shape, T}
import com.intel.analytics.bigdl.dllib.keras.ZooSpecHelper
import com.intel.analytics.bigdl.dllib.keras.serializer.ModuleSerializationTest

class SparseEmbeddingSpec extends ZooSpecHelper {

  "SparseEmbedding with one input" should "be the same as BigDL" in {

    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))

    val blayer = LookupTableSparse[Float](10, 4, "sum")
    val zlayer = SparseEmbedding[Float](inputDim = 10, outputDim = 4,
      combiner = "sum", inputShape = Shape(10))
    zlayer.build(Shape(-1, 10))

    val weight = zlayer.getWeightsBias()(0)
    blayer.weight.copy(weight)

    compareOutputAndGradInput(
      blayer.asInstanceOf[AbstractModule[Tensor[Float], Tensor[Float], Float]],
      zlayer, input, compareBackward = false)
  }

  "SparseEmbedding with two inputs" should "be the same as BigDL" in {

    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val weightValues = Array(2f, 0.5f, 1, 3)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
    val weight = Tensor.sparse(Array(indices1, indices2), weightValues, Array(3, 4))

    val blayer = LookupTableSparse[Float](10, 4, "mean")
    val zlayer = SparseEmbedding[Float](inputDim = 10, outputDim = 4,
      combiner = "mean", inputShape = Shape(10))
    zlayer.build(Shape(-1, 10))

    val zweight = zlayer.getWeightsBias()(0)
    blayer.weight.copy(zweight)

    val bout = blayer.forward(T(input, weight))
    val zout = zlayer.forward(T(input, weight))

    zout should be (bout)
  }

}

class SparseEmbeddingSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = SparseEmbedding[Float](inputDim = 10, outputDim = 4, inputShape = Shape(10))
    layer.build(Shape(-1, 10))
    val indices1 = Array(0, 0, 1, 2)
    val indices2 = Array(0, 1, 0, 3)
    val values = Array(2f, 4, 1, 2)
    val input = Tensor.sparse(Array(indices1, indices2), values, Array(3, 4))
    runSerializationTest(layer, input)
  }
}
