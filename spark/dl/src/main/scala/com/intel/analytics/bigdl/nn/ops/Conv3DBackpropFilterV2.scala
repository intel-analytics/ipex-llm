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

import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Conv3DBackpropFilterV2[T: ClassTag](
        dT: Int,
        dH: Int,
        dW: Int,
        padT: Int,
        padH: Int,
        padW: Int,
        format: DataFormat)
        (implicit ev: TensorNumeric[T])
  extends Conv3DBackpropFilter[T](dT, dH, dW, padT, padH, padW, format) {

  override protected def getParams(inputs: Table): (Int, Int, Int, Int, Int) = {
    val filterSize: Tensor[Int] = inputs[Tensor[Int]](2)

    val kT = filterSize.valueAt(1)
    val kH = filterSize.valueAt(2)
    val kW = filterSize.valueAt(3)
    val nInputPlane = filterSize.valueAt(4)
    val nOutputPlane = filterSize.valueAt(5)

    (kT, kH, kW, nInputPlane, nOutputPlane)
  }
}

object Conv3DBackpropFilterV2 {
  def apply[T: ClassTag](
                          dT: Int,
                          dH: Int,
                          dW: Int,
                          padT: Int,
                          padH: Int,
                          padW: Int,
                          format: DataFormat
                        )(implicit ev: TensorNumeric[T]): Conv3DBackpropFilterV2[T]
  = new Conv3DBackpropFilterV2[T](dT, dH, dW, padT, padH, padW, format)
}
