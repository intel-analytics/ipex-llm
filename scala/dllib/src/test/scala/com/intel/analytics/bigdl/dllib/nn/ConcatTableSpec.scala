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
class ConcatTableSpec extends FlatSpec with Matchers {

  "A ConcateTable" should "return right output and grad" in {
    val ct = new ConcatTable[Float]()
    ct.add(new Identity[Float]())
    ct.add(new Identity[Float]())

    val input = T(Tensor[Float](
      Storage(Array(1f, 2, 3))),
      T(
        Tensor[Float](Storage(Array(4f, 3, 2, 1)))
      )
    )
    val output = ct.forward(input)
    output should be (T(input, input))

    val gradOutput1 = T(
      Tensor(Storage[Float](Array(0.1f, 0.2f, 0.3f))),
      T(
        Tensor(Storage[Float](Array(0.4f, 0.3f, 0.2f, 0.1f)))
      )
    )
    val gradOutput = T(gradOutput1, gradOutput1)

    val gradInput = ct.updateGradInput(input, gradOutput)
    ct.accGradParameters(input, gradOutput)
    gradInput should be (T(
      Tensor(Storage[Float](Array(0.2f, 0.4f, 0.6f))),
      T(
        Tensor(Storage[Float](Array(0.8f, 0.6f, 0.4f, 0.2f)))
      )
    ))
  }
}
