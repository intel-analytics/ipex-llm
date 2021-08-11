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
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

@com.intel.analytics.bigdl.tags.Parallel
class PGCriterionSpec extends FlatSpec with Matchers {

  "PGCriterion " should "give correct result with dense target" in {
    val criterion = PGCriterion[Float]()

    val input = Tensor[Float](T(0.5, 0.2, 0.3))
    val target = Tensor[Float](T(1.0, 0.0, 0.0))

    criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)
    val expected = Tensor[Float](T(- 1.0/0.5 * 1.0, 0.0, 0.0))

    gradInput.almostEqual(expected, 1e-5f) should be (true)
  }

  "PGCriterion " should "give correct result with sparse target" in {
    val criterion = PGCriterion[Float]()

    val input = Tensor[Float](T(0.5, 0.2, 0.3))
    val target = Tensor.sparse(Array(Array(0)), Array(1.0f), Array(3))

    criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)
    val expected = Tensor[Float](T(- 1.0/0.5 * 1.0, 0.0, 0.0))

    gradInput.almostEqual(expected, 1e-5f) should be (true)
  }
}
