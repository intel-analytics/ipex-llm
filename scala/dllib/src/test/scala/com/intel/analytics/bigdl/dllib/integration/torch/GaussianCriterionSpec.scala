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

package com.intel.analytics.bigdl.integration.torch

import com.intel.analytics.bigdl.nn.GaussianCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}


class GaussianCriterionSpec extends FlatSpec with Matchers{
  "A GaussianCriterion Module " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val model = GaussianCriterion[Float]()

    RNG.setSeed(seed)
    val input1 = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)
    val input2 = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)
    val input = T(input1, input2)

    val target = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)

    val loss = model.forward(input, target)
    val gradInput = model.backward(input, target)

    loss should be(6.575727f +- 1e-3f)

    val gardTarget1 = Tensor(Array(-0.054713856f, 0.39738163f, -0.5449059f,
    -0.034790944f, 0.25486523f, -0.28528172f), Array(2, 3))

    val gardTarget2 = Tensor(Array(0.49651626f, 0.408394f, 0.35083658f,
    0.4992921f, 0.46332347f, 0.45096576f), Array(2, 3))

    gradInput[Tensor[Float]](1) should be(gardTarget1)
    gradInput[Tensor[Float]](2) should be(gardTarget2)
  }
}
