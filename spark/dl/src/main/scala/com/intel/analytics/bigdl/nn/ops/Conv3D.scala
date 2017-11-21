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
import com.intel.analytics.bigdl.tensor._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Table

import scala.reflect.ClassTag

class Conv3D[T: ClassTag](
                           dT: Int,
                           dH: Int,
                           dW: Int,
                           padT: Int,
                           padH: Int,
                           padW: Int
                         )(implicit ev: TensorNumeric[T]) extends Operation[Table, Tensor[T], T] {

  private val onesBias = null

  private val bias = null

  private val fInput = Tensor[T]()


  override def updateOutput(inputs: Table): Tensor[T] = {
    val input: Tensor[T] = inputs[Tensor[T]](1)
    val filter: Tensor[T] = inputs[Tensor[T]](2)

    val kT = filter.size(1)
    val kH = filter.size(2)
    val kW = filter.size(3)
    val nInputPlane = filter.size(4)
    val nOutputPlane = filter.size(5)

    val filterMM = filter.view(nInputPlane * kT * kH * kW, nOutputPlane).t()

    VolumetricConvolution.conv3d(input, output, filterMM, bias, onesBias, fInput,
      nInputPlane, nOutputPlane, withBias = false, kT, kW, kH, dT, dW, dH, padT, padW, padH)

    output
  }
}

object Conv3D {
  def apply[T: ClassTag](
                          dT: Int,
                          dH: Int,
                          dW: Int,
                          padT: Int,
                          padH: Int,
                          padW: Int
                        )(implicit ev: TensorNumeric[T]): Conv3D[T]
  = new Conv3D[T](dT, dH, dW, padT, padH, padW)
}