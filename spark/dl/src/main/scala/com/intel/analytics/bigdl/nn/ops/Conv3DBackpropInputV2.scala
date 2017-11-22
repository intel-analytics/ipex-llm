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
 (implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private val fGradInput = Tensor[T]()

  private def swap(arr: Array[Int], index1: Int, index2: Int): Array[Int] = {
    val tmp = arr(index1)
    arr(index1) = arr(index2)
    arr(index2) = tmp
    arr
  }


  override def updateOutput(inputs: Table): Tensor[T] = {
    val inputSize: Tensor[Int] = inputs[Tensor[Int]](1)
    val filter: Tensor[T] = inputs[Tensor[T]](2)
    val outputBackprop: Tensor[T] = inputs[Tensor[T]](3)

    val size = new Array[Int](5)
    var i = 0
    while (i < 5) {
      size(i) = inputSize.valueAt(i + 1)
      i = i + 1
    }
    val (transInput, transOutBackprop) = if (format == DataFormat.NHWC) {
      swap(size, 1, 4)
      swap(size, 2, 4)
      swap(size, 3, 4)
      val out = outputBackprop.transpose(2, 5).transpose(3, 5).transpose(4, 5).contiguous()
      (size, out)
    } else {
      (size, outputBackprop)
    }

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

    VolumetricConvolution.conv3DBackpropInput(transInput, output, transOutBackprop,
      weightMM, fGradInput, kT, kW, kH, dT, dW, dH, padT, padW, padH)

    if (format == DataFormat.NHWC) {
      output = output.transpose(2, 5)
      output = output.transpose(2, 3)
      output = output.transpose(3, 4)
      output = output.contiguous()
    }
    output
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