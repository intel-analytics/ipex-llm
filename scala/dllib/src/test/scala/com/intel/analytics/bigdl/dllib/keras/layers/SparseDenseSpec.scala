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

import com.intel.analytics.bigdl.dllib.nn.SparseLinear
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.Shape
import com.intel.analytics.bigdl.dllib.keras.ZooSpecHelper
import com.intel.analytics.bigdl.dllib.keras.serializer.ModuleSerializationTest

class SparseDenseSpec extends ZooSpecHelper {

  "SparseDense" should "be the same as BigDL" in {

    val blayer = SparseLinear[Float](4, 2, backwardStart = 1, backwardLength = 4)
    val zlayer = SparseDense[Float](outputDim = 2, inputShape = Shape(2, 4),
        backwardStart = 1, backwardLength = 4)
    zlayer.build(Shape(-1, 2, 4))

    val weight = zlayer.getWeightsBias()(0)
    val bias = Tensor(2)
    blayer.weight.copy(weight)
    blayer.bias.copy(bias)

    val input = Tensor[Float](Array(2, 4)).rand()
    input.setValue(1, 1, 1f)
    input.setValue(2, 3, 3f)
    val sparseInput = Tensor.sparse(input)

    compareOutputAndGradInput(blayer, zlayer, sparseInput)
  }

}

class SparseDenseSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    val layer = SparseDense[Float](outputDim = 2, inputShape = Shape(2, 4))
    layer.build(Shape(-1, 2, 4))
    val input = Tensor[Float](Array(2, 4)).rand()
    val sparseInput = Tensor.sparse(input)
    runSerializationTest(layer, sparseInput)
  }
}
