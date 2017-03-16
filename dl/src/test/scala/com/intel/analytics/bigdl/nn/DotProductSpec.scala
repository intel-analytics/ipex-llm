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
class DotProductSpec extends FlatSpec with Matchers {
  "A DotProductSpec" should "generate correct output" in {
    val input = T(
      Tensor[Float](Storage(Array(1f, 2, 3))),
      Tensor[Float](Storage(Array(4f, 5, 6)))
    )

    val gradOutput = Tensor(Storage[Float](Array(8.9f)))

    val expectedOutput = Tensor(Storage[Float](Array(32f)))

    val expectedgradInput = T(
      Tensor(Storage[Float](Array(35.6f, 44.5f, 53.4f))),
      Tensor(Storage[Float](Array(8.9f, 17.8f, 26.7f)))
    )

    val dot = new DotProduct[Float]()

    val dotOutput = dot.forward(input)
    val dotGradInput = dot.backward(input, gradOutput)

    dotOutput should be (expectedOutput)
    dotGradInput should be (expectedgradInput)
  }
}
