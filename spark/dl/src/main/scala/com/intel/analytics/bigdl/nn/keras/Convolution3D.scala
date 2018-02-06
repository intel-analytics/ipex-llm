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

package com.intel.analytics.bigdl.nn.keras

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, TensorModule}
import com.intel.analytics.bigdl.nn.{InitializationMethod, VolumetricConvolution, Xavier, Zeros}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class Convolution3D[T: ClassTag](
   val nbFilter: Int,
   val kernelDim1: Int,
   val kernelDim2: Int,
   val kernelDim3: Int,
   val init: InitializationMethod = Xavier,
   val activation: AbstractModule[Tensor[T], Tensor[T], T] = null,
   val borderMode: String = "valid",
   val subsample: Array[Int] = Array(1, 1, 1),
   val wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(borderMode == "valid" || borderMode == "same", s"Invalid border mode for " +
    s"Convolution3D: $borderMode")
  require(subsample.length == 3,
    s"For Convolution3D, subsample should be of length 3 but got length ${subsample.length}")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val pads = KerasUtils.getPadsFromBorderMode3D(borderMode)
    val layer = VolumetricConvolution(
      nInputPlane = input(1),
      nOutputPlane = nbFilter,
      kT = kernelDim1,
      kW = kernelDim3,
      kH = kernelDim2,
      dT = subsample(0),
      dW = subsample(2),
      dH = subsample(1),
      padT = pads._1,
      padW = pads._3,
      padH = pads._2,
      withBias = bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer)
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    KerasLayer.fuse(layer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Convolution3D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    kernelDim1: Int,
    kernelDim2: Int,
    kernelDim3: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    borderMode: String = "valid",
    subsample: Array[Int] = Array(1, 1, 1),
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Convolution3D[T] = {
    new Convolution3D[T](nbFilter, kernelDim1, kernelDim2, kernelDim3,
      KerasUtils.getInitMethod(init), KerasUtils.getActivation(activation), borderMode, subsample,
      wRegularizer, bRegularizer, bias, inputShape)
  }
}
