/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.pipeline.api.keras.layers

import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, InferShape}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, SingleShape, T}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

abstract class LayerWrapperByForward[T: ClassTag](
      val batchInputShape: Shape)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](batchInputShape) with Net {

  override def computeOutputShape(calcInputShape: Shape): Shape = {
    LayerWrapperByForward.computeOutputShape[T](doBuild(calcInputShape), calcInputShape)
  }
}

private[zoo] object LayerWrapperByForward {

  private def singleShapeDummyValue[T: ClassTag](
     singleShape: Shape)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    Tensor[T](
      (List(2) ++ KerasUtils.removeBatch(singleShape).toSingle()).toArray).fill(ev.one)
  }

  def computeOutputShape[T: ClassTag](torchLayer: AbstractModule[Activity, Activity, T],
                         calcInputShape: Shape)(implicit ev: TensorNumeric[T]): Shape = {
    val input: Activity = calcInputShape match {
      case s: SingleShape => singleShapeDummyValue(s)
      case m: MultiShape =>
        T.array(m.toMulti().map(singleShapeDummyValue(_)).toArray)
    }
    val dummyOutTensor = torchLayer.cloneModule().forward(input)
    require(dummyOutTensor.isTensor, "We only support single output for now but got a Table")
    val outSize = dummyOutTensor.toTensor.size()
    KerasUtils.addBatch(Shape(outSize.slice(1, outSize.length)))
  }
}

/**
 * Wrap a torch style layer to keras style layer.
 * This layer can be built multiple times.
 * @param torchLayer a torch style layer
 *   i.e If the input data is (2, 3, 4) and 2 is the batch size, you should input: (3, 4) here.
 * @return a keras compatible layer
 */
class KerasLayerWrapper[T: ClassTag]
(val torchLayer: AbstractModule[Activity, Activity, T],
 val inputShape: Shape = null)(implicit ev: TensorNumeric[T])
  extends LayerWrapperByForward[T](KerasUtils.addBatch(inputShape)) {
  setName(torchLayer.getName() + "_wrapper")
  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = torchLayer
}
