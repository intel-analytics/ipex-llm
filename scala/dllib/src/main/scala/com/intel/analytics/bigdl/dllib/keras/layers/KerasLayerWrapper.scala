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

package com.intel.analytics.bigdl.dllib.keras.layers

import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils.{MultiShape, Shape, SingleShape, T}
import com.intel.analytics.bigdl.dllib.keras.Net
import com.intel.analytics.bigdl.dllib.keras.layers.utils.KerasUtils

import scala.reflect.ClassTag

abstract class LayerWrapperByForward[T: ClassTag](
      val batchInputShape: Shape)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](batchInputShape) with Net {

  override def clearState() : this.type = {
    if (output.isInstanceOf[Tensor[_]]) {
      output = Tensor[T]()
    }

    if (gradInput.isInstanceOf[Tensor[_]]) {
      gradInput = Tensor[T]()
    }

    this
  }

  override def computeOutputShape(calcInputShape: Shape): Shape = {
    LayerWrapperByForward.computeOutputShape[T](doBuild(calcInputShape), calcInputShape)
  }
}

private[bigdl] object LayerWrapperByForward {

  private def singleShapeDummyValue[T: ClassTag](
     singleShape: Shape)(implicit ev: TensorNumeric[T]): Tensor[T] = {
    val newShape = new Array[Int](singleShape.toSingle().length)
    singleShape.toSingle().copyToArray(newShape)
    for (i <- 0 until newShape.length) {
      if (newShape(i) == -1) {
        newShape(i) = 1
      }
    }
    Tensor[T](newShape).fill(ev.one)
  }

  private def shapeDummyValue[T: ClassTag](shape: Shape)
    (implicit ev: TensorNumeric[T]): Activity = {
    if (shape.isInstanceOf[SingleShape]) return singleShapeDummyValue(shape)
    T.array(shape.toMulti().map(shapeDummyValue(_)).toArray)
  }

  private def isConcreteShape(shape: Shape): Boolean = {
    if (shape.isInstanceOf[SingleShape]) return shape.toSingle().forall(_ > 0)
    shape.toMulti().forall(isConcreteShape(_))
  }

  private def getShape[T: ClassTag](output: Activity, addBatch: Boolean)
                                   (implicit ev: TensorNumeric[T]): Shape = {
    var shape: Shape = null
    if (output.isTensor) {
      val outSize = output.toTensor[T].size()
      shape = if (!addBatch) {
        Shape(outSize)
      } else {
        KerasUtils.addBatch(Shape(outSize.slice(1, outSize.length)))
      }
      return shape
    }
    MultiShape(output.toTable.toSeq[Activity].map(d => getShape(d, addBatch)).toList)
  }

  def computeOutputShape[T: ClassTag](torchLayer: AbstractModule[Activity, Activity, T],
                         calcInputShape: Shape)(implicit ev: TensorNumeric[T]): Shape = {
    val input = shapeDummyValue(calcInputShape)
    val dummyOutTensor = torchLayer.cloneModule().forward(input)
    getShape(dummyOutTensor, !isConcreteShape(calcInputShape))
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
