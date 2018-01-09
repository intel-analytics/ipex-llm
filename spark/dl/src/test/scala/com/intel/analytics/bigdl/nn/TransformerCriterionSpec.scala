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
class TransformerCriterionSpec extends FlatSpec with Matchers {

  "TransformerCriterion" should "work correctly" in {

    import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric.NumericFloat

    val criterion = TransformerCriterion[Float](MSECriterion[Float](),
      Some(Square[Float]()), Some(Square[Float]()))

    val input = Tensor(1, 3, 224, 224).rand()
    val target = Tensor(1, 3, 224, 224).rand()

    val loss = criterion.forward(input, target)
    val gradInput = criterion.backward(input, target)

    val squaredInput = Tensor(1, 3, 224, 224).copy(input).square()
    val squaredTarget = Tensor(1, 3, 224, 224).copy(target).square()

    val referenceCriterion = MSECriterion()
    val expectedLoss = referenceCriterion.forward(squaredInput, squaredTarget)
    val expectedGradInput = referenceCriterion
      .backward(squaredInput, squaredTarget).cmul(input).mul(2.0f)

    loss should be (expectedLoss)
    gradInput should be (expectedGradInput)
  }
}
