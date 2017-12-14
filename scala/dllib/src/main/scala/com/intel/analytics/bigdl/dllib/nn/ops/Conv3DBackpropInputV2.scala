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

import com.intel.analytics.bigdl.nn.VolumetricConvolution
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Conv3DBackpropInputV2[T: ClassTag](dT: Int, dH: Int, dW: Int,
                                        padT: Int, padH: Int, padW: Int,
                                         format: DataFormat)
 (implicit ev: TensorNumeric[T])
  extends Conv3DBackpropInput[T](dT, dH, dW, padT, padH, padW, format) {

  private val fGradInput = Tensor[T]()

  override protected def getInputSize(inputs: Table): Array[Int] = {
    val inputSize: Tensor[Int] = inputs[Tensor[Int]](1)

    if (format == DataFormat.NHWC) {
      val N = inputSize.valueAt(1)
      val D = inputSize.valueAt(2)
      val H = inputSize.valueAt(3)
      val W = inputSize.valueAt(4)
      val C = inputSize.valueAt(5)
      Array(N, C, D, H, W)
    } else {
      val N = inputSize.valueAt(1)
      val C = inputSize.valueAt(2)
      val D = inputSize.valueAt(3)
      val H = inputSize.valueAt(4)
      val W = inputSize.valueAt(5)
      Array(N, C, D, H, W)
    }
  }

  override def clearState(): Conv3DBackpropInputV2.this.type = {
    super.clearState()
    fGradInput.set()
    this
  }
}

object Conv3DBackpropInputV2 {
  def apply[T: ClassTag](
                          dT: Int,
                          dH: Int,
                          dW: Int,
                          padT: Int,
                          padH: Int,
                          padW: Int,
                          format: DataFormat
                        )(implicit ev: TensorNumeric[T]): Conv3DBackpropInputV2[T]
  = new Conv3DBackpropInputV2[T](dT, dH, dW, padT, padH, padW, format)
}
