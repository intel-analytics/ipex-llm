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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}


class KLDCriterionSpec extends FlatSpec with Matchers{
  "A KLDCriterion Module " should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val model = KLDCriterion[Float]()

    RNG.setSeed(seed)
    val input1 = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)
    val input2 = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)
    val input = T(input1, input2)

    val target = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)

    val loss = model.forward(input, target)
    val gradInput = model.backward(input, target)

    loss should be(0.991884f / 2 +- 1e-3f)

    val gardTarget1 = Tensor(Array(0.54340494f, 0.67115563f, 0.2783694f,
    0.4120464f, 0.4245176f, 0.52638245f), Array(2, 3)).mul(0.5f)

    val gardTarget2 = Tensor(Array(0.66372836f, 0.08010721f, 0.002364993f,
    0.084828794f, 0.06463373f, 0.10249251f), Array(2, 3)).mul(0.5f)

    gradInput[Tensor[Float]](1) should be(gardTarget1)
    gradInput[Tensor[Float]](2) should be(gardTarget2)
  }

  "A KLDCriterion Module with standard normal input" should "generate correct output and grad" in {
    val seed = 100
    RNG.setSeed(seed)
    val model = KLDCriterion[Float]()

    RNG.setSeed(seed)
    val input1 = Tensor[Float](2, 3).fill(0.0f)
    val input2 = Tensor[Float](2, 3).fill(0.0f)
    val input = T(input1, input2)

    val target = Tensor[Float](2, 3).apply1(x => RNG.uniform(0, 1).toFloat)

    val loss = model.forward(input, target)
    val gradInput = model.backward(input, target)

    loss should be(0.0f)

    val gardTarget1 = Tensor[Float](2, 3).fill(0.0f)

    val gardTarget2 = Tensor[Float](2, 3).fill(0.0f)

    gradInput[Tensor[Float]](1) should be(gardTarget1)
    gradInput[Tensor[Float]](2) should be(gardTarget2)
  }
}
