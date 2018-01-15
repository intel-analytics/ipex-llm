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

import com.intel.analytics.bigdl.nn.abstractnn.{TensorCriterion, TensorModule}
import com.intel.analytics.bigdl.tensor.{DenseType, SparseType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Convert DenseTensor to SparseTensor.
 * @param propagateBack whether propagate gradient back, default value is true
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class DenseToSparse[T: ClassTag](val propagateBack: Boolean = true // propagate gradient back
                                )(implicit ev: TensorNumeric[T]) extends TensorModule[T] {
  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    require(input.getTensorType == DenseType, "DenseToSparse: input should be a DenseTensor," +
      s"but got ${input.getTensorType}")
    output = Tensor.sparse(input)
    output
  }
  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    if (propagateBack) {
      this.gradInput.resizeAs(input)
      Tensor.dense(gradOutput, gradInput)
    }
    this.gradInput
  }

  override def toString(): String = s"DenseToSparse()"
}

object DenseToSparse {
  def apply[@specialized(Float, Double) T: ClassTag]
  (propagateBack: Boolean = true)(implicit ev: TensorNumeric[T]) : DenseToSparse[T] = {
    new DenseToSparse(propagateBack)
  }
}
