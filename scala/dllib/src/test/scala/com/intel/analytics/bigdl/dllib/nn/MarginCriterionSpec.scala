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
import org.scalatest.{FlatSpec, Matchers}


class MarginCriterionSpec extends FlatSpec with Matchers {

  "MarginCriterion " should "calculate correct squared hinge loss" in {
    val input = Tensor[Float](Array[Float](0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    val target = Tensor[Float](Array[Float](0.4f, 0.3f, 0.2f, 0.1f), Array(4))

    val criterion = MarginCriterion[Float](squared = true)

    val loss = criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)
    val expectedGradInput =
      Tensor[Float](Array[Float](-0.192f, -0.141f, -0.094f, -0.048f), Array(4))

    math.abs(loss - 0.9026) < 1e-5 should be (true)

    gradInput.almostEqual(expectedGradInput, 1e-5) should be (true)

  }

  "MarginCriterion " should "calculate correct squared hinge loss 2" in {
    val input = Tensor[Float](Array[Float](1f, 0.2f, 3f, 0.4f), Array(4))
    val target = Tensor[Float](Array[Float](4f, 0.3f, 2f, 0.1f), Array(4))

    val criterion = MarginCriterion[Float](squared = true)

    val loss = criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)
    val expectedGradInput =
      Tensor[Float](Array[Float](-0.0f, -0.141f, -0.0f, -0.048f), Array(4))

    math.abs(loss - 0.4513) < 1e-5 should be (true)

    gradInput.almostEqual(expectedGradInput, 1e-5) should be (true)

  }

  "MarginCriterion " should "calculate correct hinge loss" in {
    val input = Tensor[Float](Array[Float](0.1f, 0.2f, 0.3f, 0.4f), Array(4))
    val target = Tensor[Float](Array[Float](0.4f, 0.3f, 0.2f, 0.1f), Array(4))

    val criterion = MarginCriterion[Float](sizeAverage = false)

    val loss = criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)
    val expectedGradInput =
      Tensor[Float](Array[Float](-0.4f, -0.3f, -0.2f, -0.1f), Array(4))

    math.abs(loss - 3.8) < 1e-5 should be (true)

    gradInput.almostEqual(expectedGradInput, 1e-5) should be (true)

  }

}
