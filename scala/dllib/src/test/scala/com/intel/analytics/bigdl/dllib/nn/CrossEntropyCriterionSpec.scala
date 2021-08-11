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

import scala.math._


  /**
  * Created by ywan on 16-9-21.
  */

@com.intel.analytics.bigdl.tags.Parallel
class CrossEntropyCriterionSpec extends FlatSpec with Matchers {

  "CrossEntropyCriterion " should "return return right output and gradInput" in {
    val criterion = new CrossEntropyCriterion[Double]()

    val input = Tensor[Double](3, 3)
    input(Array(1, 1)) = 0.33655226649716
    input(Array(1, 2)) = 0.77367000770755
    input(Array(1, 3)) = 0.031494265655056
    input(Array(2, 1)) = 0.11129087698646
    input(Array(2, 2)) = 0.14688249188475
    input(Array(2, 3)) = 0.49454387230799
    input(Array(3, 1)) = 0.45682632108219
    input(Array(3, 2)) = 0.85653987620026
    input(Array(3, 3)) = 0.42569971177727

    val target = Tensor[Double](3)
    target(Array(1)) = 1
    target(Array(2)) = 2
    target(Array(3)) = 3

    val expectedOutput = 1.2267281042702334

    val loss = criterion.forward(input, target)
    loss should be(expectedOutput +- 1e-8)

    val expectedGrad = Tensor[Double](3, 3)
    expectedGrad(Array(1, 1)) = -0.23187185
    expectedGrad(Array(1, 2)) = 0.15708656
    expectedGrad(Array(1, 3)) = 0.07478529
    expectedGrad(Array(2, 1)) = 0.09514888
    expectedGrad(Array(2, 2)) = -0.23473696
    expectedGrad(Array(2, 3)) = 0.13958808
    expectedGrad(Array(3, 1)) = 0.09631823
    expectedGrad(Array(3, 2)) = 0.14364876
    expectedGrad(Array(3, 3)) = -0.23996699
    val gradInput = criterion.backward(input, target)

    expectedGrad.map(gradInput, (v1, v2) => {
      assert(abs(v1 - v2) < 1e-6);
      v1
    })

  }
}
