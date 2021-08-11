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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.nn.{InferReshape => Reshape}
import org.scalatest.{FlatSpec, Matchers}

class ReshapeSpec extends FlatSpec with Matchers {
  "Reshape operation" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    val input =
      Tensor(T(
        T(
          T(1f, 1f, 1f),
          T(2f, 2f, 2f)
        ),
        T(
          T(3f, 3f, 3f),
          T(4f, 4f, 4f)
        ),
        T(
          T(5f, 5f, 5f),
          T(6f, 6f, 6f)
        )
      ))

    val expectOutput1 = Tensor(T(
      T(1f, 1f, 1f, 2f, 2f, 2f, 3f, 3f, 3f),
      T(4f, 4f, 4f, 5f, 5f, 5f, 6f, 6f, 6f)))
    val expectOutput2 = Tensor(T(
      T(1f, 1f, 1f, 2f, 2f, 2f, 3f, 3f, 3f),
      T(4f, 4f, 4f, 5f, 5f, 5f, 6f, 6f, 6f)))
    val expectOutput3 = Tensor(T(
      T(
        T(1f, 1f, 1f),
        T(2f, 2f, 2f),
        T(3f, 3f, 3f)
      ),
      T(
        T(4f, 4f, 4f),
        T(5f, 5f, 5f),
        T(6f, 6f, 6f)
      )
    ))

    val output1 = Reshape(Array(2, -1)).forward(input)
    val output2 = Reshape(Array(-1, 9)).forward(input)
    val output3 = Reshape(Array(2, -1, 3)).forward(input)

    output1 should be(expectOutput1)
    output2 should be(expectOutput2)
    output3 should be(expectOutput3)
  }
}
