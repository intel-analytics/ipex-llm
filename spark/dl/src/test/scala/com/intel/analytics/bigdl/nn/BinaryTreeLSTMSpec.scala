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
package com.intel.analytics.bigdl.torch

import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.T
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class BinaryTreeLSTMSpec extends FlatSpec with Matchers with BeforeAndAfter {
  "BinaryTreeLSTM" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericDouble
    import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

    RNG.setSeed(100)

    val hiddenSize = 2
    val inputSize = 2

    val inputs =
      Tensor(
        T(T(T(1.0, 2.0),
          T(2.0, 3.0),
          T(4.0, 5.0))))

    val tree =
      Tensor(
        T(T(T(2.0, 5.0, -1.0),
          T(0.0, 0.0, 1.0),
          T(0.0, 0.0, 2.0),
          T(0.0, 0.0, 3.0),
          T(3.0, 4.0, 0.0))))

    val input = T(inputs, tree)

    val gradOutput =
      Tensor(
        T(T(T(2.0, 5.0),
          T(2.0, 3.0),
          T(4.0, 5.0),
          T(2.0, 3.0),
          T(4.0, 5.0),
          T(6.0, 7.0))))

    val expectOutput =
      Tensor(
        T(T(T(-0.07799374051859737, -0.14419464399333934),
          T(-0.2349552348774636, -0.04679071771123799),
          T(-0.1594515102098235, -0.026039638054106272),
          T(-0.04540739978946999, -0.0070662412123771254),
          T(-0.05869603467391258, -0.13559056761784405))))

    val expectGradInputs =
      Tensor(
        T(T(
          T(0.5614597104995146, -0.3383652016018004),
          T(0.8172036851171792, -0.46767634057453855),
          T(0.37739630380493044, -0.2335553148048936))))

    val expectGradTree =
      Tensor(
        T(T(
          T(0.0, 0.0, 0.0),
          T(0.0, 0.0, 0.0),
          T(0.0, 0.0, 0.0),
          T(0.0, 0.0, 0.0),
          T(0.0, 0.0, 0.0))))

    val expectGradInput = T(expectGradInputs, expectGradTree)

    val model = BinaryTreeLSTM(inputSize, hiddenSize)

    val output = model.forward(input)
    println(output)
    output should be(expectOutput)

    val gradInput = model.backward(input, gradOutput)
    println(gradInput)
    gradInput should be(expectGradInput)
  }
}
