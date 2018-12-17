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

import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

/**
 * Computes the maximum of elements across dimensions of a tensor.
 * The input of Max should be two tensor, the first one is data,
 * the second one is the dimension to compute maximum.
 * @param keepDims if keepDims is false, will delete the singleton dimension
 *                 in output.
 * @param startFromZero if the dimension count from zero.
 */
class Max[T: ClassTag, D: ClassTag](
        keepDims: Boolean = false,
        startFromZero: Boolean = false
      )(implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]
      ) extends Operation[Table, Tensor[D], T] {

  output = Tensor[D]()
  // just a buffer using in tensor.max.
  protected val indices: Tensor[D] = Tensor[D]()

  override def updateOutput(input: Table): Tensor[D] = {
    val x = input[Tensor[D]](1)
    val y = input[Tensor[Int]](2)

    require(y.isScalar || (y.nElement() == 1 && y.dim() == 1),
      s"reduction indices should be a scalar or one-element tensor")
    val reductionIndices = if (startFromZero) {
      y.value() + 1
    } else {
      y.value()
    }
    require(reductionIndices <= x.nDimension(), s"reduction indices should smaller than" +
      s" input's dimension, excepted smaller than ${x.dim()}, but got ${reductionIndices}")

    x.max(output, indices, reductionIndices)

    if(keepDims) {
      output
    } else {
      output.squeeze(reductionIndices)
    }
  }

  override def getClassTagNumerics() : (Array[ClassTag[_]], Array[TensorNumeric[_]]) = {
    (Array[ClassTag[_]](scala.reflect.classTag[T], scala.reflect.classTag[D]),
      Array[TensorNumeric[_]](ev, ev2))
  }

  override def clearState(): Max.this.type = {
    super.clearState()
    indices.set()
    this
  }
}

object Max {
  def apply[T: ClassTag, D: ClassTag](
    keepDims: Boolean = false,
    startFromZero: Boolean = false)(
      implicit ev: TensorNumeric[T], ev2: TensorNumeric[D]): Max[T, D] = {
    new Max[T, D](keepDims, startFromZero)
  }
}
