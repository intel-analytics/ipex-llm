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

import com.intel.analytics.bigdl.nn.VariableFormat.OUT_IN_KT_KH_KW
import com.intel.analytics.bigdl.nn.VolumetricConvolution
import com.intel.analytics.bigdl.nn.abstractnn.DataFormat
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Conv3DBackpropFilter[T: ClassTag](
                                        dT: Int,
                                        dH: Int,
                                        dW: Int,
                                        padT: Int,
                                        padH: Int,
                                        padW: Int,
                                        format: DataFormat
       )(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private val fInput = Tensor[T]()


  protected def getParams(inputs: Table): (Int, Int, Int, Int, Int) = {
    val filter: Tensor[T] = inputs[Tensor[T]](2)

    val kT = filter.size(1)
    val kH = filter.size(2)
    val kW = filter.size(3)
    val nInputPlane = filter.size(4)
    val nOutputPlane = filter.size(5)

    (kT, kH, kW, nInputPlane, nOutputPlane)
  }
  override def updateOutput(inputs: Table): Tensor[T] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)
    val outputBackprop: Tensor[T] = inputs[Tensor[T]](3)

    val (transInput, transOutBackprop) = if (format == DataFormat.NHWC) {
      // backpropInput only use input size, so we do not need it to be contiguous
      val in = input.transpose(2, 5).transpose(3, 5).transpose(4, 5).contiguous()
      val out = outputBackprop.transpose(2, 5).transpose(3, 5).transpose(4, 5).contiguous()
      (in, out)
    } else {
      (input, outputBackprop)
    }

    val (kT, kH, kW, nInputPlane, nOutputPlane) = getParams(inputs)

    val gradWeightMM = Tensor[T](nOutputPlane, nInputPlane * kT * kH * kW)

    VolumetricConvolution.populateFInput(transInput, fInput, nInputPlane, nOutputPlane,
      kT, kW, kH, dT, dW, dH, padT, padW, padH)

    VolumetricConvolution.conv3DBackpropFilter(transInput, transOutBackprop, gradWeightMM,
      null, fInput, 1.0, 1.0, false)

    output = if (format == DataFormat.NHWC) {
      val gradWeight = gradWeightMM.view(nOutputPlane, nInputPlane, kT, kH, kW)
      gradWeight.transpose(1, 5).transpose(2, 4).transpose(1, 3).contiguous()
    } else {
      gradWeightMM.view(nOutputPlane, nInputPlane, kT, kH, kW)
    }

    output
  }

  override def clearState(): Conv3DBackpropFilter.this.type = {
    super.clearState()
    fInput.set()
    this
  }
}

object Conv3DBackpropFilter {
  def apply[T: ClassTag](
                          dT: Int,
                          dH: Int,
                          dW: Int,
                          padT: Int,
                          padH: Int,
                          padW: Int,
                          format: DataFormat
                        )(implicit ev: TensorNumeric[T]): Conv3DBackpropFilter[T]
  = new Conv3DBackpropFilter[T](dT, dH, dW, padT, padH, padW, format)
}
