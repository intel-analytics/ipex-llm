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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.nn.CAddTable
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class AddSpec extends FlatSpec with Matchers {
  "Add operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      T(
        Tensor(T(1f, 2f, 3f)),
        Tensor(T(2f, 2f, 4f)),
        Tensor(T(7f, 3f, 1f))
      )

    val expectOutput = Tensor(T(10f, 7f, 8f))

    val output = CAddTable().forward(input)
    output should be(expectOutput)
  }
}
