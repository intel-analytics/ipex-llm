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

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat, TensorModule}
import com.intel.analytics.bigdl.optim.Regularizer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class LocallyConnected2D[T: ClassTag](val nbFilter: Int,
                                      val nbRow: Int,
                                      val nbCol: Int,
                                      val subsample: (Int, Int) = (1, 1),
                                      val borderMode: String = "valid",
                                      var wRegularizer: Regularizer[T] = null,
                                      var bRegularizer: Regularizer[T] = null,
                                      val bias: Boolean = true,
                                      val format: DataFormat = DataFormat.NCHW,
                                      val activation: TensorModule[T] = null,
                                      var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(borderMode.toLowerCase() == "valid" || borderMode.toLowerCase() == "same",
    s"$borderMode is not supported")

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val stackSize = if (format == DataFormat.NCHW) input(1) else input(3)
    val inputWidth = if (format == DataFormat.NCHW) input(3) else input(2)
    val inputHeight = if (format == DataFormat.NCHW) input(2) else input(1)
    val pad = KerasUtils.getPadsFromBorderMode(borderMode)

    val layer = com.intel.analytics.bigdl.nn.LocallyConnected2D(
      nInputPlane = stackSize,
      inputWidth = inputWidth,
      inputHeight = inputHeight,
      nOutputPlane = nbFilter,
      kernelW = nbCol,
      kernelH = nbRow,
      strideW = subsample._2,
      strideH = subsample._1,
      padW = pad._1,
      padH = pad._2,
      wRegularizer = wRegularizer,
      bRegularizer = bRegularizer,
      withBias = bias,
      format = format
    )
    KerasLayer.fuse(layer, activation,
      inputShape).asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object LocallyConnected2D {
  def apply[@specialized(Float, Double) T: ClassTag](
    nbFilter: Int,
    nbRow: Int,
    nbCol: Int,
    subsample: (Int, Int) = (1, 1),
    borderMode: String = "valid",
    wRegularizer: Regularizer[T] = null,
    bRegularizer: Regularizer[T] = null,
    bias: Boolean = true,
    format: DataFormat = DataFormat.NCHW,
    activation: String = null,
    inputShape: Shape = null)
    (implicit ev: TensorNumeric[T]): LocallyConnected2D[T] = {
    new LocallyConnected2D[T](
      nbFilter,
      nbRow,
      nbCol,
      subsample,
      borderMode,
      wRegularizer,
      bRegularizer,
      bias,
      format,
      KerasUtils.getActivation(activation),
      inputShape)
  }
}
