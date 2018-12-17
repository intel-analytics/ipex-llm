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
import com.intel.analytics.bigdl.utils.RandomGenerator._
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleSerializationTest
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class BinaryTreeLSTMSpec extends FlatSpec with Matchers with BeforeAndAfter {
  "BinaryTreeLSTM" should "works correctly" in {
    import com.intel.analytics.bigdl.numeric.NumericFloat
    import com.intel.analytics.bigdl.utils.RandomGenerator.RNG

    RNG.setSeed(100)

    val hiddenSize = 2
    val inputSize = 2

    val inputs =
      Tensor(
        T(T(T(1f, 2f),
          T(2f, 3f),
          T(4f, 5f))))

    val tree =
      Tensor(
        T(T(T(2f, 5f, -1f),
          T(0f, 0f, 1f),
          T(0f, 0f, 2f),
          T(0f, 0f, 3f),
          T(3f, 4f, 0f))))

    val input = T(inputs, tree)

    val gradOutput =
      Tensor(
        T(T(T(2f, 5f),
          T(2f, 3f),
          T(4f, 5f),
          T(2f, 3f),
          T(4f, 5f),
          T(6f, 7f))))

    val expectOutput =
      Tensor(
        T(T(T(-0.07799374051859737f, -0.14419464399333934f),
          T(-0.2349552348774636f, -0.04679071771123799f),
          T(-0.1594515102098235f, -0.026039638054106272f),
          T(-0.04540739978946999f, -0.0070662412123771254f),
          T(-0.05869603467391258f, -0.13559056761784405f))))

    val expectGradInputs =
      Tensor(
        T(T(
          T(0.5614597104995146f, -0.3383652016018004f),
          T(0.8172036851171792f, -0.46767634057453855f),
          T(0.37739630380493044f, -0.2335553148048936f))))

    val expectGradTree =
      Tensor(
        T(T(
          T(0f, 0f, 0f),
          T(0f, 0f, 0f),
          T(0f, 0f, 0f),
          T(0f, 0f, 0f),
          T(0f, 0f, 0f))))

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

class BinaryTreeLSTMSerialTest extends ModuleSerializationTest {
  override def test(): Unit = {
    RNG.setSeed(1000)
    val binaryTreeLSTM = BinaryTreeLSTM[Float](2, 2).setName("binaryTreeLSTM")

    val inputs =
      Tensor[Float](
        T(T(T(1f, 2f),
          T(2f, 3f),
          T(4f, 5f))))

    val tree =
      Tensor[Float](
        T(T(T(2f, 5f, -1f),
          T(0f, 0f, 1f),
          T(0f, 0f, 2f),
          T(0f, 0f, 3f),
          T(3f, 4f, 0f))))

    val input = T(inputs, tree)

    runSerializationTest(binaryTreeLSTM, input)
  }
}

