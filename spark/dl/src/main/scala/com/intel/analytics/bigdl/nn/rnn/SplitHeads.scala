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
package com.intel.analytics.bigdl.nn.rnn

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath._

import scala.reflect.ClassTag

/**
 * Split x into different heads, and transpose the resulting value.
 * The tensor is transposed to insure the inner dimensions hold the correct
 * values during the matrix multiplication.
 * input with shape [batch_size, length, hidden_size]
 * output with shape [batch_size, num_heads, length, hidden_size/num_heads]
 * @param hiddenSize
 * @param numHeads
 * @param mul
 * @tparam T The numeric type in this module parameters
 */
private[nn] class SplitHeads[T: ClassTag](hiddenSize: Int, numHeads: Int,
                              mul: Boolean = false)(implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  private val depth = hiddenSize / numHeads
  private val value = ev.fromType(math.pow(depth, -0.5))
  private val permutations: (Int, Int) = (2, 3)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val batchSize = input.size(1)
    val length = input.size(2)

    output.resizeAs(input).copy(input)
    output = output.reshape(Array(batchSize, length, numHeads, depth))
      .transpose(permutations._1, permutations._2)
    if (mul) {
      output.mul(value)
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (mul) {
      gradInput.resizeAs(gradOutput).zero().add(value, gradOutput)
    } else {
      gradInput.resizeAs(gradOutput).copy(gradOutput)
    }
    gradInput = gradInput.transpose(permutations._1, permutations._2).contiguous()
    gradInput.resize(input.size())
    gradInput
  }
}
