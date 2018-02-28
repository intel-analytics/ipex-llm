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

import com.intel.analytics.bigdl.tensor.{SparseType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.reflect.ClassTag

/**
 * Indicator operation represents multi-hot representation of given Tensor.
 *
 * The Input Tensor should be a 2-D Sparse Tensor.
 * And used to transform the output tensor of CategoricalCol* ops.
 *
 * The output tensor should be a DenseTensor with shape (batch, feaLen).
 *
 * For example, A input SparseTensor as follows:
 *  indices(0) = Array(0, 0, 1, 2, 2)
 *  indices(1) = Array(0, 3, 1, 1, 2)
 *  values     = Array(1, 2, 2, 3, 3)
 *  shape      = Array(3, 4)
 *
 *  the output tensor should be an 2D 3x4 DenseTensor with isCount = true
 *  0.0, 1.0, 1.0, 0.0
 *  0.0, 0.0, 1.0, 0.0
 *  0.0, 0.0, 0.0, 2.0
 *
 * @param feaLen The length of feature.
 * @param isCount whether count the times of same value, default: true.
 * @tparam T Numeric type. Parameter tensor numeric type. Only support float/double now
 */

class IndicatorCol[T: ClassTag](
  val feaLen: Int,
  val isCount: Boolean = true
) (implicit ev: TensorNumeric[T])
  extends Operation[Tensor[Int], Tensor[T], T]{

  output = Tensor[T]()

  override def updateOutput(input: Tensor[Int]): Tensor[T] = {

    require(input.getTensorType == SparseType, "Only sparse input is supported")

    val rows = input.size(dim = 1)
    val resTensor = Tensor[T](rows, feaLen)

    var i = 1
    while (i <= rows) {
      val narrowTensor = input.narrow(1, i, 1)
      val tempArr = narrowTensor.storage().array().slice(
        narrowTensor.storageOffset()-1, narrowTensor.storageOffset() - 1 + narrowTensor.nElement())
      var j = 0
      while (j < tempArr.length) {
        require(tempArr(j) < feaLen, "the parameter feaLen is set too small")
        isCount match {
          case false =>
            resTensor.setValue(i, tempArr(j) + 1, ev.one)
          case true =>
            val res = ev.toType[Int](resTensor.valueAt(i, tempArr(j) + 1)) + 1
            resTensor.setValue(i, tempArr(j) + 1, ev.fromType[Int](res))
        }
        j += 1
      }
      i += 1
    }
    output = resTensor
    output
  }
}

object IndicatorCol {
  def apply[T: ClassTag](
    feaLen: Int,
    isCount: Boolean = true
  )(implicit ev: TensorNumeric[T]): IndicatorCol[T]
  = new IndicatorCol[T](
    feaLen = feaLen,
    isCount = isCount
  )
}
