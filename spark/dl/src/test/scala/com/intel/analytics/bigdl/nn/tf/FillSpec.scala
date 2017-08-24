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
package com.intel.analytics.bigdl.nn.tf

import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{FlatSpec, Matchers}

class FillSpec extends FlatSpec with Matchers {

  "Fill forward" should "be correct" in {
    val layer = Fill(0.1)
    val shape = Tensor(T(2.0f, 3.0f))
    layer.forward(shape) should be(Tensor(T(T(0.1f, 0.1f, 0.1f), T(0.1f, 0.1f, 0.1f))))
  }

  "Fill backward" should "be correct" in {
    val layer = Fill(0.1)
    val shape = Tensor(T(2.0f, 3.0f))
    val gradOutput = Tensor(2, 3).rand()
    layer.forward(shape) should be(Tensor(T(T(0.1f, 0.1f, 0.1f), T(0.1f, 0.1f, 0.1f))))
    layer.backward(shape, gradOutput) should be(Tensor(T(0.0f, 0.0f)))
  }
}
