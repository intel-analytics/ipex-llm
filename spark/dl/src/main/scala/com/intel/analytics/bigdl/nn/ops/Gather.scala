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
package com.intel.analytics.bigdl.nn.ops

import com.intel.analytics.bigdl.tensor.{IntType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Gather slices from first input tensor according to the second input tensor.
 * Input should be two tensors, the first one is the tensor which to gather values;
 * the second one is Index tensor.
 */
class Gather[T: ClassTag, D: ClassTag](
  var dim: Int = 1)(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D])
  extends Operation[Table, Tensor[D], T]{
  output = Tensor[D]()

  protected val intBuffer = Tensor[Int]()

  override def updateOutput(input: Table): Tensor[D] = {
    val inputTensor = input[Tensor[D]](1)
    val input2 = input[Tensor[_]](2)
    // support floatType indices.
    val indices = if (input2.getType() == IntType) {
      input2.asInstanceOf[Tensor[Int]]
    } else {
      intBuffer.resizeAs(input2)
      input2.cast[Int](intBuffer)
      intBuffer
    }
    val inputSizes = inputTensor.size()
    val inputDim = inputTensor.dim() // data batch dim
    dim = if (dim <= 0) {
      inputDim + dim
    }
    else dim
    require(dim >= 1 && dim <= inputDim, s"Invalid position: $dim. " +
      s"input:dim() is $inputTensor, input feature map dim (numInputDims) is $inputDim.")

    // set output shape
    val indicesSize = indices.size()
    val outputSizes = if (indices.isScalar) {
      inputSizes.slice(0, dim-1) ++ Array(1) ++ inputSizes.slice(dim, inputSizes.length)
    } else {
      inputSizes.slice(0, dim-1) ++ indicesSize ++ inputSizes.slice(dim, inputSizes.length)
    }
    // set the insert position in output to one-dim array
    output.resize(inputSizes.slice(0, dim-1)++
      Array(indices.nElement())++
      inputSizes.slice(dim, inputSizes.length))

    // copy selected element to the insert position
    indices.resize(indices.nElement())
    var i = 0
    while (i < indices.nElement()) {
      val index = indices.valueAt(i + 1)
      require(index < inputSizes(dim - 1),
        s"index should smaller than ${inputSizes(dim - 1)}, but got $index")
      output.select(dim, i + 1).copy(inputTensor.select(dim, index + 1))
      i += 1
    }
    // resize the output to expected shape
    indices.resize(indicesSize)
    output.resize(outputSizes)
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }

  override def clearState() : this.type = {
    super.clearState()
    intBuffer.set()
    this
  }

}

object Gather {
  def apply[T: ClassTag, D: ClassTag](
    dim: Int = 1
  )(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]):
  Gather[T, D] = new Gather(dim)
}
