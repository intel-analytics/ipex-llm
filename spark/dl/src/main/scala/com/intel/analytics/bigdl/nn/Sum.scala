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

import com.intel.analytics.bigdl.nn.abstractnn.TensorModule
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * It is a simple layer which applies a sum operation over the given dimensions.
 * When nInputDims is provided, the input will be considered as a batches.
 * Then the sum operation will be applied in (dimension + 1)
 *
 * The input to this layer is expected to be a tensor, or a batch of tensors;
 * when using mini-batch, a batch of sample tensors will be passed to the layer and
 * the user need to specify the number of dimensions of each sample tensor in the
 * batch using `nInputDims`.
 *
 * @param dimensions the dimensions to be applied sum operation
 * @param nInputDims specify the number of dimensions that this module will receive
 *                   If it is more than the dimension of input tensors, the first dimension
 *                   would be considered as batch size
 * @param sizeAverage default is false, if it is true, it will return the mean instead
 * @param keepSize default is false, if it is true, it will keep the sum dimensions, for example
 *               if it is false, sum 2*2*2 on dimension 3 will be 2*2
 *               while if it is true the result size will be 2*2*1
 */

@SerialVersionUID(- 8025422596092583688L)
class Sum[T: ClassTag](
    dimensions: Array[Int] = Array(1),
    nInputDims: Int = -1,
    sizeAverage: Boolean = false,
    keepSize: Boolean = false)
    (implicit ev: TensorNumeric[T]) extends TensorModule[T] {

  override def updateOutput(input: Tensor[T]): Tensor[T] = {
    getPositiveDimension(input)
    output.resizeAs(input).copy(input)
    var i = 0
    while(i < _dims.length) {
      val dim = _dims(i)
      output = output.sum(dim)
      if (sizeAverage) {
        output.div(ev.fromType[Int](input.size(dim)))
      }
      if (!keepSize) {
        output.squeeze(dim)
      }
      i += 1
    }
    output
  }

  override def updateGradInput(input: Tensor[T], gradOutput: Tensor[T]): Tensor[T] = {
    // the size array is cloned, so modifaction on size won't affect input tensor
    val size = input.size()
    var i = 0
    while(i < _dims.size) {
      size(_dims(i) - 1) = 1
      i += 1
    }

    if (gradOutput.isContiguous()) {
      _gradOutput = gradOutput.view(size)
    } else {
      _gradOutput = gradOutput.contiguous().view(size)
    }

    gradInput.resizeAs(input)
    gradInput.copy(_gradOutput.expandAs(input))
    if (sizeAverage) {
      _dims.foreach(dimension => gradInput.div(ev.fromType[Int](input.size(dimension))))
    }
    gradInput
  }

  override def toString: String = s"nn.Sum"

  private val distinctDims = dimensions.distinct

  private val _dims = new Array[Int](distinctDims.length)

  @transient
  private var _gradOutput: Tensor[T] = null

  private def getPositiveDimension(input: Tensor[T]): Unit = {
    System.arraycopy(distinctDims, 0, _dims, 0, distinctDims.length)
    var i = 0
    while(i < distinctDims.length) {
      if (_dims(i) < 0) {
        _dims(i) = input.dim() + _dims(i) + 1
      }
      if (nInputDims > 0 && input.dim() == (nInputDims + 1)) {
        _dims(i) += 1
      }
      require(input.dim() >= _dims(i), "dimension exceeds input dimensions")
      i += 1
    }
  }
}

object Sum {
  def apply[T: ClassTag](
      dimension: Int = 1,
      nInputDims: Int = -1,
      sizeAverage: Boolean = false)(implicit ev: TensorNumeric[T]): Sum[T] = {
    new Sum[T](Array(dimension), nInputDims, sizeAverage)
  }

  def apply[T: ClassTag](
      dimension: Array[Int],
      nInputDims: Int,
      sizeAverage: Boolean,
      keepSize: Boolean)(implicit ev: TensorNumeric[T]) : Sum[T] = {
    new Sum[T](dimension, nInputDims, sizeAverage, keepSize)
  }
}
