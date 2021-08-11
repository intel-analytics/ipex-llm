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

@com.intel.analytics.bigdl.tags.Parallel
class ParallelCriterionSpec extends FlatSpec with Matchers {
  "A ParallelCriterion" should "generate correct output with type Double" in {
    val pc = new ParallelCriterion[Double]()

    val input = T(Tensor[Double](2, 10), Tensor[Double](2, 10))
    var i = 0
    input[Tensor[Double]](1).apply1(_ => {i += 1; i})
    input[Tensor[Double]](2).apply1(_ => {i -= 1; i})
    val target = T(Tensor[Double](Storage(Array(1.0, 8.0))), Tensor[Double](2, 10).fill(1.0))
    val nll = new ClassNLLCriterion[Double]()
    val mse = new MSECriterion[Double]()
    pc.add(nll, 0.5).add(mse)
    val output = pc.forward(input, target)
    val gradInput = pc.backward(input, target)
    output should be (100.75)
  }

  "A ParallelCriterion" should "generate correct output with type Float" in {
    val pc = new ParallelCriterion[Float]()

    val input = T(Tensor[Float](2, 10), Tensor[Float](2, 10))
    var i = 0
    input[Tensor[Float]](1).apply1(_ => {i += 1; i})
    input[Tensor[Float]](2).apply1(_ => {i -= 1; i})
    val target = T(Tensor[Float](Storage(Array(1.0f, 8.0f))), Tensor[Float](2, 10).fill(1.0f))
    val nll = new ClassNLLCriterion[Float]()
    val mse = new MSECriterion[Float]()
    pc.add(nll, 0.5).add(mse)
    val output = pc.forward(input, target)
    val gradInput = pc.backward(input, target)
    output should be (100.75)
  }

}
