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

import com.intel.analytics.bigdl.nn.{InitializationMethod, Ones, SpatialBatchNormalization, Zeros}
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, DataFormat}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape

import scala.reflect.ClassTag

class BatchNormalization[T: ClassTag](
   val epsilon: Double = 0.001,
   val momentum: Double = 0.99,
   val betaInit: Tensor[T] = null,
   val gammaInit: Tensor[T] = null,
   val dimOrdering: String = "th",
   var inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Tensor[T], Tensor[T], T](KerasLayer.addBatch(inputShape)) {

  require(dimOrdering.toLowerCase() == "th" || dimOrdering.toLowerCase() == "tf",
    s"Dim ordering must be either tf or th, but got ${dimOrdering.toLowerCase()}")
  private val format = dimOrdering.toLowerCase() match {
    case "th" => DataFormat.NCHW
    case "tf" => DataFormat.NHWC
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    val input = inputShape.toSingle().toArray
    require(input.length == 4,
      s"BatchNormalization requires 4D input, but got input dim ${input.length}")
    inputShape
  }

  override def doBuild(inputShape: Shape): AbstractModule[Tensor[T], Tensor[T], T] = {
    val input = inputShape.toSingle().toArray
    val nChannel = format match {
      case DataFormat.NCHW => input(1)
      case DataFormat.NHWC => input(3)
    }
    // TODO: support arbitrary input shape
    val layer = SpatialBatchNormalization(
      nOutput = nChannel,
      eps = epsilon,
      momentum = momentum,
      initWeight = gammaInit,
      initBias = betaInit,
      dataFormat = format
    )
    layer.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object BatchNormalization {
  def apply[@specialized(Float, Double) T: ClassTag](
    epsilon: Double = 0.001,
    momentum: Double = 0.99,
    betaInit: Tensor[T] = null,
    gammaInit: Tensor[T] = null,
    format: String = "th",
    inputShape: Shape = null)(implicit ev: TensorNumeric[T]): BatchNormalization[T] = {
    new BatchNormalization[T](epsilon, momentum, betaInit, gammaInit, format, inputShape)
  }
}
