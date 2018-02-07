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

import com.intel.analytics.bigdl.tensor.{DoubleType, FloatType, Tensor}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

/**
 * Indicator operation represents multi-hot representation of given Tensor.
 *
 * The Input Tensor only support Sparse Tensor.
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
  feaLen: Int,
  isCount: Boolean = true
) (implicit ev: TensorNumeric[T])
  extends Operation[Tensor[Int], Tensor[T], T]{

  output = Tensor[T]()

  override def updateOutput(input: Tensor[Int]): Tensor[T] = {
    val rows = input.size(dim = 1)
    val indices0 = new ArrayBuffer[Int]()
    val indices1 = new ArrayBuffer[Int]()
    val values = new ArrayBuffer[T]()
    val indexMap = mutable.HashMap[Int, Int]()
    var i = 1
    while (i <= rows) {
      val narrowTensor = input.narrow(1, i, 1)
      val selectedArr = narrowTensor.storage().array().slice(
        narrowTensor.storageOffset()-1, narrowTensor.storageOffset() - 1 + narrowTensor.nElement()
      )
      selectedArr.foreach { x =>
        if (!indexMap.contains(x)) {
          indexMap(x) = 1
        }
        else {
          indexMap(x) = indexMap(x) + 1
        }
      }
      indexMap.foreach { kv =>
        indices0 += i-1
        indices1 += kv._1
        ev.getType() match {
          case DoubleType =>
            if (isCount) {
              values += kv._2.toDouble.asInstanceOf[T]
            }
            else {
              values += 1.toDouble.asInstanceOf[T]
            }
          case FloatType =>
            if (isCount) {
              values += kv._2.toFloat.asInstanceOf[T]
            }
            else {
              values += 1.toFloat.asInstanceOf[T]
            }
        }
      }
      i += 1
      indexMap.clear()
    }
    val indices = Array(indices0.toArray, indices1.toArray)
    val shape = Array(rows, feaLen)
    output = Tensor.dense(
      Tensor.sparse(indices, values.toArray, shape))
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
