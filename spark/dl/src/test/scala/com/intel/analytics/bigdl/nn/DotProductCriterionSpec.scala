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

@com.intel.analytics.bigdl.tags.Parallel
class DotProductCriterionSpec extends FlatSpec with Matchers {

  "DotProductCriterion " should "give correct result with dense target" in {
    val criterion = DotProductCriterion[Float]()

    val input = Tensor[Float](Array(4, 6)).rand()
    val target = input.clone()

    val loss = criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)

    loss should be (input.sumSquare() +- 1e-5f)
    gradInput.almostEqual(target, 1e-5f) should be (true)
  }

  "DotProductCriterion " should "give correct result with sparse target" in {
    val criterion = DotProductCriterion[Float]()

    val input = Tensor[Float](Array(4, 6)).rand()
    val target = Tensor.sparse(input.clone())

    val loss = criterion.forward(input, target)

    val gradInput = criterion.backward(input, target)

    loss should be (input.sumSquare() +- 1e-5f)
    gradInput.almostEqual(Tensor.dense(target), 1e-5f) should be (true)
  }
}
