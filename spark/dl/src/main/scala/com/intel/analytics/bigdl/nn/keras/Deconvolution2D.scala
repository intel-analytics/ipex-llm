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
import com.intel.analytics.bigdl.nn.{InitializationMethod, SpatialFullConvolution, Xavier, Zeros}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class Deconvolution2D[T: ClassTag](
   val nbFilter: Int,
   val nbRow: Int,
   val nbCol: Int,
   val init: InitializationMethod = Xavier,
   val activation: TensorModule[T] = null,
   val subsample: (Int, Int) = (1, 1),
   var wRegularizer: Regularizer[T] = null,
   var bRegularizer: Regularizer[T] = null,
   val bias: Boolean = true,
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val layer = SpatialFullConvolution(
      nInputPlane = input(1),
      nOutputPlane = nbFilter,
      kW = nbCol,
      kH = nbRow,
      dW = subsample._2,
      dH = subsample._1,
      noBias = !bias,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer
    )
    layer.setInitMethod(weightInitMethod = init, biasInitMethod = Zeros)
    KerasLayer.fuse(layer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object Deconvolution2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    init: String = "glorot_uniform",
    activation: String = null,
    subsample: (Int, Int) = (1, 1),
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): Deconvolution2D[T] = {
    new Deconvolution2D[T](nbFilter, nbRow, nbCol, KerasUtils.getInitMethod(init),
      KerasUtils.getActivation(activation), subsample,
      wRegularizer, bRegularizer, bias, inputShape)
  }
}