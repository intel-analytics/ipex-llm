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

import scala.reflect.ClassTag

/**
 * MkString operation converts a SparseTensor/DenseTensor to a Dense Tensor[String]
 *
 * the output shape will be 1-D Tensor[String].
 *
 * @param strDelimiter The delimiter between values, default: ","
 * @tparam T Numeric type. Parameter tensor numeric type. Only support float/double now
 */

class MkString[T: ClassTag](
  val strDelimiter: String = ","
)(implicit ev: TensorNumeric[T])
  extends Operation[Tensor[_], Tensor[String], T]{
  output = Tensor[String]()
  override def updateOutput(input: Tensor[_]): Tensor[String] = {

    val rows = input.size(dim = 1)
    val resTensor = Tensor[String](rows)
    var i = 1
    while (i <= rows) {
      val narrowTensor = input.narrow(1, i, 1)
      val resStr = narrowTensor.storage().array().slice(
        narrowTensor.storageOffset() - 1,
        narrowTensor.storageOffset() -1 + narrowTensor.nElement()
      ).mkString(strDelimiter)

      resTensor.setValue(i, resStr)
      i += 1
    }
    output = resTensor
    output
  }
}

object MkString {
  def apply[T: ClassTag](
    strDelimiter: String = ","
  ) (implicit ev: TensorNumeric[T]): MkString[T]
    = new MkString[T](
    strDelimiter = strDelimiter
  )
}
