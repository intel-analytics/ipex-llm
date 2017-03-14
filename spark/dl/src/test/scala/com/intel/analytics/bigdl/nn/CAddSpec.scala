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

import com.intel.analytics.bigdl.nn.abstractnn.{TensorCriterion, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.utils.RandomGenerator._

@com.intel.analytics.bigdl.tags.Parallel
class CAddSpec extends FlatSpec with Matchers {

  "A CAdd(5, 1)" should "should converge" in {
    val seed = 100
    RNG.setSeed(seed)

    val layer = new CAdd[Float](Array(5, 1))
    val mse = new MSECriterion[Float]()
    val y = Tensor[Float](5, 4)
    val bf = Tensor[Float](5, 4)
    for (i <- 1 to 5) {
      bf(i).fill(i)
    }

    def gradUpdate(mlp : TensorModule[Float], x : Tensor[Float], y : Tensor[Float],
      criterion : TensorCriterion[Float], learningRate : Float) : Float = {

      val pred = mlp.forward (x)
      val err = criterion.forward (pred, y)
      val gradCriterion = criterion.backward (pred, y)
      mlp.zeroGradParameters ()
      mlp.backward (x, gradCriterion)
      mlp.updateParameters (learningRate)
      err
    }

    for (i <- 1 to 10000) {
      val x = Tensor.randperm[Float](20)
      x.resize(5, 4)
      y.copy(x)
      y.add(bf)
      val err = gradUpdate(layer, x, y, mse, 0.1f)
    }

    layer.bias(Array(1, 1)) should be(1.0f +- 1e-4f)
    layer.bias(Array(2, 1)) should be(2.0f +- 1e-4f)
    layer.bias(Array(3, 1)) should be(3.0f +- 1e-4f)
    layer.bias(Array(4, 1)) should be(4.0f +- 1e-4f)
    layer.bias(Array(5, 1)) should be(5.0f +- 1e-4f)
  }
}
