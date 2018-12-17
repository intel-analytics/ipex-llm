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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * First split the tensor along the [[dimension]] into [[numSplit]] sub tensors,
 * then select the [[index]]th one
 */
@SerialVersionUID(-9096120159559947483L)
private[bigdl] class SplitAndSelect[T: ClassTag]
  (val dimension: Int, val index: Int, val numSplit: Int)
                         (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    val dim = if (dimension < 0) input.dim() + dimension + 1 else dimension
    val dimSize = input.size(dimension)
    require(dimSize % numSplit == 0,
      s"numSplit must evenly divides input.size(dimension), " +
        s"numSplit: $numSplit, dimension: $dimension, dimSize: $dimSize")
    val length = dimSize / numSplit
    val offset = (index - 1) * length + 1
    val outputNarrow = input.narrow(dim, offset, length)
    output.resizeAs(outputNarrow).copy(outputNarrow)
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    val dim = if (dimension < 0) input.dim() + dimension + 1 else dimension
    val dimSize = input.size(dimension)
    val length = dimSize / numSplit
    val offset = (index - 1) * length + 1
    gradInput.resizeAs(input).zero()
    gradInput.narrow(dim, offset, length).copy(gradOutput)
    gradInput
  }
}

private[bigdl] object SplitAndSelect {
  def apply[T: ClassTag](
      dimension: Int,
      index: Int,
      numSplit: Int)(implicit ev: TensorNumeric[T]) : SplitAndSelect[T] = {
    new SplitAndSelect[T](dimension, index, numSplit)
  }
}
