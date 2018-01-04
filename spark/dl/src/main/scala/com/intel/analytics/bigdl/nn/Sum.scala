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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * It is a simple layer which applies a sum operation over the given dimension.
 * When nInputDims is provided, the input will be considered as a batches.
 * Then the sum operation will be applied in (dimension + 1)
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 * when using mini-batch, a batch of sample tensors will be passed to the layer and
 * the user need to specify the number of dimensions of each sample tensor in the
 * batch using `nInputDims`.
 *
 * @param dimension the dimension to be applied sum operation
 * @param nInputDims specify the number of dimensions that this module will receive
 *                   If it is more than the dimension of input tensors, the first dimension
 *                   would be considered as batch size
 * @param sizeAverage default is false, if it is true, it will return the mean instead
 * @param squeeze default is true, which will squeeze the sum dimension; set it to false to keep
 *                the sum dimension
 */

@SerialVersionUID(- 8025422596092583688L)
class Sum[T: ClassTag, D: ClassTag](
  private var dimension: Int = 1,
  nInputDims: Int = -1,
  sizeAverage: Boolean = false,
  squeeze: Boolean = true)
  (implicit ev: TensorNumeric[T], evd: TensorNumeric[D])
  extends AbstractModule[Tensor[D], Tensor[D], T] {
  @transient
  private var _gradOutput: Tensor[D] = null

  output = Tensor[D]()
  gradInput = Tensor[D]()

  private def getPositiveDimension(input: Tensor[D]): Int = {
    var dimension = this.dimension
    if (dimension < 0) {
      dimension = input.dim() + dimension + 1
    }

    if (nInputDims > 0 && input.dim() == (nInputDims + 1)) {
      dimension += 1
    }

    require(input.dim() >= dimension, "dimension exceeds input dimensions" +
      s"dimension $dimension, input dimension ${input.dim()}")
    dimension
  }

  def changeSumDims(d: Int): this.type = {
    dimension = d
    this
  }

  override def updateOutput(input: Tensor[D]): Tensor[D] = {
    val dimension = getPositiveDimension(input)
    output.sum(input, dimension)

    if (sizeAverage) {
      output.div(evd.fromType(input.size(dimension)))
    }

    if (output.nDimension() > 1 && squeeze) {
      output.squeeze(dimension)
    }

    if (output.nElement() == 1 && squeeze) {
      output = Tensor.scalar[D](output.storage.apply(output.storageOffset() - 1))
    }

    output
  }

  override def updateGradInput(input: Tensor[D], gradOutput: Tensor[D]): Tensor[D] = {
    val dimension = getPositiveDimension(input)
    val size = input.size()
    size(dimension - 1) = 1

    if (!gradOutput.isContiguous()) {
      _gradOutput = gradOutput.clone().view(size)
    } else {
      _gradOutput = gradOutput.view(size)
    }
    gradInput.resizeAs(input)
    gradInput.copy(_gradOutput.expandAs(input))
    if (sizeAverage) {
      gradInput.div(evd.fromType(input.size(dimension)))
    }
    gradInput
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, evd))
  }

  override def toString: String = s"nn.Sum"
}

object Sum {
  def apply[T: ClassTag, D: ClassTag](
    dimension: Int = 1,
    nInputDims: Int = -1,
    sizeAverage: Boolean = false,
    squeeze: Boolean = true)(implicit ev: TensorNumeric[T], evd: TensorNumeric[D]) : Sum[T, D] = {
    new Sum[T, D](dimension, nInputDims, sizeAverage, squeeze)
  }
}
