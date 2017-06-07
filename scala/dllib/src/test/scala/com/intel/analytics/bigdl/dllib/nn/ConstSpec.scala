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

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T

class ConstSpec extends FlatSpec with Matchers {
  "Const forward tensor" should "be correct" in {
    val value = Tensor(2, 3).rand()
    val layer = Const(value)
    val input = Tensor(4, 5).rand()
    layer.forward(input) should be(value)
  }

  "Const backward tensor" should "be correct" in {
    val value = Tensor(2, 3).rand()
    val layer = Const(value)
    val input = Tensor(4, 5).rand()
    val gradOutput = Tensor(2, 3).rand()
    layer.forward(input) should be(value)
    val grad = layer.backward(input, gradOutput).toTensor
    grad should be(Tensor(4, 5).zero())
  }

  "Const forward tensors" should "be correct" in {
    val value = Tensor(2, 3).rand()
    val layer = Const(value)
    val input = T(Tensor(4, 5).rand(), Tensor(3, 4).rand())
    layer.forward(input) should be(value)
  }

  "Const backward tensor" should "be correct when input is tensors" in {
    val value = Tensor(2, 3).rand()
    val layer = Const(value)
    val input = T(Tensor(4, 5).rand(), Tensor(3, 4).rand())
    val gradOutput = Tensor(2, 3).rand()
    layer.forward(input) should be(value)
    val grad = layer.backward(input, gradOutput).toTable
    grad[Tensor[Float]](1) should be(Tensor(4, 5).zero())
    grad[Tensor[Float]](2) should be(Tensor(3, 4).zero())
  }
}
