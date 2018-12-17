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

import com.intel.analytics.bigdl.nn.abstractnn.AbstractCriterion
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * This is same with cross entropy criterion, except the target tensor is a one-hot tensor
 *
 * @tparam T The numeric type in the criterion, usually which are [[Float]] or [[Double]]
 */
class CategoricalCrossEntropy[T: ClassTag]()(implicit ev: TensorNumeric[T])
  extends AbstractCriterion[Tensor[T], Tensor[T], T]{

  private val crxEntropy = CrossEntropyCriterion[T]()

  import CategoricalCrossEntropy._

  private val buffer = Tensor[T]()

  override def updateOutput(input: Tensor[T], target: Tensor[T]): T = {
    buffer.resizeAs(input)
    crxEntropy.forward(buffer.log(input), convertTensor(target))
  }

  override def backward(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput = crxEntropy.backward(buffer, convertTensor(target))
    gradInput.div(input)
  }

  override def updateGradInput(input: Tensor[T], target: Tensor[T]): Tensor[T] = {
    gradInput = crxEntropy.updateGradInput(buffer, convertTensor(target))
    gradInput.div(input)
  }
}

object CategoricalCrossEntropy {
  def apply[T: ClassTag]()(implicit ev: TensorNumeric[T]): CategoricalCrossEntropy[T] =
    new CategoricalCrossEntropy()

  private def convertTensor[T](tensor: Tensor[T]): Tensor[T] = {
    tensor.max(2)._2
  }
}
