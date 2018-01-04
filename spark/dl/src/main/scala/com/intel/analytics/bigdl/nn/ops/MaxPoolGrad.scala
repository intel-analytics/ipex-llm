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

import com.intel.analytics.bigdl.nn.SpatialMaxPooling
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class MaxPoolGrad[T: ClassTag](
  kH: Int,
  kW: Int,
  strideW: Int,
  strideH: Int,
  padH: Int,
  padW: Int,
  format: DataFormat
)(implicit ev: TensorNumeric[T])
  extends Operation[Table, Tensor[T], T]{

  private var module : SpatialMaxPooling[T] = _

  override def updateOutput(input: Table): Tensor[T] = {
    if (module == null) {
      module = SpatialMaxPooling[T](
        kH,
        kW,
        strideH,
        strideW,
        padH,
        padW,
        format
      )
    }

    val inputData = input[Tensor[T]](1)
    val gradOutput = input[Tensor[T]](3)
    module.updateOutput(inputData)
    output = module.updateGradInput(inputData, gradOutput)
    output
  }
}

object MaxPoolGrad {
  def apply[T: ClassTag](
    kH: Int,
    kW: Int,
    strideW: Int,
    strideH: Int,
    padH: Int,
    padW: Int,
    format: DataFormat
  )(implicit ev: TensorNumeric[T]): MaxPoolGrad[T] =
    new MaxPoolGrad(kH, kW, strideW, strideH, padH, padW, format)
}
