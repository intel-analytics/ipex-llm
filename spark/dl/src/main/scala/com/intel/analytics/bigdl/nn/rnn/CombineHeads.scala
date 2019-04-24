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
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

// Combine tensor that has been split.
//  input should be tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
// output should be tensor with shape [batch_size, length, hidden_size]
private[nn] class CombineHeads[T: ClassTag](implicit ev: TensorNumeric[T])
  extends TensorModule[T] {

  private val permutations: (Int, Int) = (2, 3)

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val batchSize = input.size(1)
    val length = input.size(3)
    val hiddenSize = input.size(2) * input.size(4)

    output.resizeAs(input).copy(input)
    output = output.transpose(permutations._1, permutations._2)
      .reshape(Array(batchSize, length, hiddenSize))
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val size = Array(input.size(1), input.size(3), input.size(2), input.size(4))
    if (gradOutput.isContiguous()) {
      gradInput = gradOutput.view(size)
    } else {
      gradInput = gradOutput.contiguous().view(size)
    }
    gradInput = gradInput.transpose(permutations._1, permutations._2).contiguous()
    gradInput
  }
}
