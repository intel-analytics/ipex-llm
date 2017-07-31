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
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Serial
class DiceCoefficientCriterionSpec extends FlatSpec with Matchers {

  "A DiceCoefficientCriterionSpec" should "generate correct output and gradInput vector input" in {
    val input = Tensor[Float](Storage[Float](Array(0.1f, 0.2f)))
    val target = Tensor[Float](Storage[Float](Array(0.2f, 0.3f)))

    val expectedOutput = 0.35555553f

    val expectedgradInput =
      Tensor[Float](Storage[Float](Array(0.13580247f, 0.02469136f)))

    val criterion = DiceCoefficientCriterion[Float](epsilon = 1.0f)
    val loss = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)

    loss should be (expectedOutput +- 1e-5f)
    gradInput.map(expectedgradInput, (a, b) => {
      a should be (b +- 1e-5f)
      a
    })
  }

  "A DiceCoefficientCriterionSpec" should "generate correct output and gradInput scala input" in {
    val input = Tensor[Float](Storage[Float](Array(0.1f)))
    val target = Tensor[Float](Storage[Float](Array(0.2f)))

    val expectedOutput = 0.2f

    val expectedgradInput =
      Tensor[Float](Storage[Float](Array(0.30769231f)))

    val criterion = DiceCoefficientCriterion[Float](epsilon = 1.0f)
    val loss = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)

    loss should be (expectedOutput +- 1e-5f)
    gradInput.map(expectedgradInput, (a, b) => {
      a should be (b +- 1e-5f)
      a
    })
  }

  "A DiceCoefficientCriterionSpec" should "generate correct output and gradInput batch input" in {
    val input = Tensor[Float](Storage[Float](Array(0.3f, 0.8f, 0.7f, 1.3f)),
      storageOffset = 1, size = Array(2, 2))
    val target = Tensor[Float](Storage[Float](Array(1.5f, 2.5f, 3.5f, 4.5f)),
      storageOffset = 1, size = Array(2, 2))

    val expectedOutput = -0.28360658884f

    val expectedgradInput =
      Tensor[Float](Storage[Float](Array(-0.16662188f, -0.33055633f, -0.24545453f, -0.3363636f)),
        storageOffset = 1, size = Array(2, 2))

    val criterion = DiceCoefficientCriterion[Float](epsilon = 1.0f)
    val loss = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)

    loss should be (expectedOutput +- 1e-5f)
    gradInput.map(expectedgradInput, (a, b) => {
      a should be (b +- 1e-5f)
      a
    })
  }

  "A DiceCoefficientCriterionSpec" should "generate pass gradient check" in {
    val input = Tensor[Float](Array(3, 3, 3)).rand
    val target = Tensor[Float](Array(3, 3, 3)).rand

    val criterion = DiceCoefficientCriterion[Float](epsilon = 0.1f)

    println("gradient check for input")
    val gradCheckerInput = new GradientChecker(1e-2, 1)
    val checkFlagInput = gradCheckerInput.checkCriterion[Float](criterion, input, target)
  }
}
