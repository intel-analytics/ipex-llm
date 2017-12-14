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

class Conv3DBackpropInput[T: ClassTag](
                           dT: Int,
                           dH: Int,
                           dW: Int,
                           padT: Int,
                           padH: Int,
                           padW: Int,
                           format: DataFormat
                         )(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private val fGradInput = Tensor[T]()

  protected def getInputSize(inputs: Table): Array[Int] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)

    if (format == DataFormat.NHWC) {
      val N = input.size(1)
      val D = input.size(2)
      val H = input.size(3)
      val W = input.size(4)
      val C = input.size(5)
      Array(N, C, D, H, W)
    } else {
      val N = input.size(1)
      val C = input.size(2)
      val D = input.size(3)
      val H = input.size(4)
      val W = input.size(5)
      Array(N, C, D, H, W)
    }
  }

  override def updateOutput(inputs: Table): Tensor[T] = {

    val filter: Tensor[T] = inputs[Tensor[T]](2)
    val outputBackprop: Tensor[T] = inputs[Tensor[T]](3)

    val transOutBackprop = if (format == DataFormat.NHWC) {
      // backpropInput only use input size, so we do not need it to be contiguous
      outputBackprop.transpose(2, 5).transpose(3, 5).transpose(4, 5).contiguous()
    } else {
      outputBackprop
    }

    val transInputSize = getInputSize(inputs)

    val kT = filter.size(1)
    val kH = filter.size(2)
    val kW = filter.size(3)
    val nInputPlane = filter.size(4)
    val nOutputPlane = filter.size(5)

    var transWeight = filter.transpose(1, 5)
    transWeight = transWeight.transpose(2, 4)
    transWeight = transWeight.transpose(3, 5)
    transWeight = transWeight.contiguous()
    val weightMM = transWeight.view(nOutputPlane, nInputPlane * kT * kH * kW)

    VolumetricConvolution.conv3DBackpropInput(transInputSize, output, transOutBackprop,
      weightMM, fGradInput, kT, kW, kH, dT, dW, dH, padT, padW, padH)

    if (format == DataFormat.NHWC) {
      output = output.transpose(2, 5)
      output = output.transpose(2, 3)
      output = output.transpose(3, 4)
      output = output.contiguous()
    }
    output
  }

  override def clearState(): Conv3DBackpropInput.this.type = {
    super.clearState()
    fGradInput.set()
    this
  }
}

object Conv3DBackpropInput {
  def apply[T: ClassTag](
                          dT: Int,
                          dH: Int,
                          dW: Int,
                          padT: Int,
                          padH: Int,
                          padW: Int,
                          format: DataFormat
                        )(implicit ev: TensorNumeric[T]): Conv3DBackpropInput[T]
  = new Conv3DBackpropInput[T](dT, dH, dW, padT, padH, padW, format)
}
