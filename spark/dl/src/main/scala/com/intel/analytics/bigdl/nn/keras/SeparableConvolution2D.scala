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

import com.intel.analytics.bigdl.nn.{InitializationMethod, SpatialSeperableConvolution, Xavier}
import com.intel.analytics.bigdl.nn.abstractnn._
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}

import scala.reflect.ClassTag

class SeparableConvolution2D[T: ClassTag](val nbFilter: Int,
                                          val nbCol: Int,
                                          val nbRow: Int,
                                         // val init: InitializationMethod = Xavier,
                                        //  val activation: TensorModule[T] = null,
                                          val borderMode: String = "valid",
                                          val subsample: (Int, Int) = (1, 1),
                                          val depthMultiplier: Int = 1,
                                          val bias: Boolean = true,
                                          val format: DataFormat = DataFormat.NCHW,
                                          var depthwiseRegularizer: Regularizer[T] = null,
                                          var bRegularizer: Regularizer[T] = null,
                                          var pointwiseRegularizer: Regularizer[T] = null,
                                          var inputShape: Shape = null
  )(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(borderMode.toLowerCase() == "valid" || borderMode.toLowerCase() == "same",
    s"$borderMode is not supported")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val stackSize = if (format == DataFormat.NCHW) input(1) else input(3)
    val pad = KerasUtils.getPadsFromBorderMode(borderMode)

    val layer = SpatialSeperableConvolution(
      nInputChannel = stackSize,
      nOutputChannel = nbFilter,
      depthMultiplier = depthMultiplier,
      kW = nbCol,
      kH = nbRow,
      sW = subsample._2,
      sH = subsample._1,
      pW = pad._2,
      pH = pad._1,
      hasBias = bias,
      dataFormat = format,
      wRegularizer = depthwiseRegularizer,
      bRegularizer = bRegularizer,
      pRegularizer = pointwiseRegularizer
    )
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
//    KerasLayer.fuse(layer, activation,
//      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}


object SeparableConvolution2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbCol: Int,
    nbRow: Int,
//    init: InitializationMethod = Xavier,
//    activation: String = null,
    borderMode: String = "valid",
    subsample: (Int, Int) = (1, 1),
    depthMultiplier: Int = 1,
    bias: Boolean = true,
    format: DataFormat = DataFormat.NCHW,
    depthwiseRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    pointwiseRegularizer: Regularizer[T] = null,
    inputShape: Shape = null
    )(implicit ev: TensorNumeric[T]) : SeparableConvolution2D[T] = {
    new SeparableConvolution2D[T](
      nbFilter,
      nbCol,
      nbRow,
//      init,
//      KerasUtils.getActivation(activation),
      borderMode,
      subsample,
      depthMultiplier,
      bias,
      format,
      depthwiseRegularizer,
      bRegularizer,
      pointwiseRegularizer,
      inputShape)
  }
}
