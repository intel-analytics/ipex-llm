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

package com.intel.analytics.bigdl.nn.abstractnn

import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.utils.{Node, Shape}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

trait InferShape {

  private[bigdl] var _inputShapeValue: Shape = null

  private[bigdl] var _outputShapeValue: ArrayBuffer[Shape] = ArrayBuffer[Shape]()

  private[bigdl] def inputShapeValue: Shape = _inputShapeValue

  private[bigdl] def inputShapeValue_=(value: Shape): Unit = {
    _inputShapeValue = value
  }

  private[bigdl] def outputShapeValue: ArrayBuffer[Shape] = _outputShapeValue

  private[bigdl] def outputShapeValue_=(value: ArrayBuffer[Shape]): Unit = {
    _outputShapeValue = value
  }

  private[bigdl] def compatibleWithKeras(): Boolean = true

  private[bigdl] def compatibleWithTorch(): Boolean = true

  private[bigdl] def excludeNotTorch[T: ClassTag]
  (nodes : Seq[Node[AbstractModule[_, _, T]]]): Unit = {
    val invalidNodes = nodes.filter{!_.element.compatibleWithTorch()}
    if (invalidNodes.length > 0) {
      throw new RuntimeException(s"Do not mix with Layer: ${invalidNodes.mkString(",")}")
    }
  }

  private[bigdl] def excludeNotKeras[T: ClassTag]
  (nodes : Seq[ModuleNode[T]]): Unit = {
      val invalidNodes = nodes.filter{!_.element.compatibleWithKeras()}
      if (invalidNodes.length > 0) {
        throw new RuntimeException(s"Do not mix with Layer: ${invalidNodes.mkString(",")}")
      }
  }

  /**
   * There's no batch dim in the inputShape which just represent a sample record.
   */
  private[bigdl] def getInputShape(): Shape = {
    _inputShapeValue
  }

  /**
   * Get the outputshape by index.
   * @param index start from 0
   * @return
   */
  private[bigdl] def getOutputShapeFor(index: Int): Shape = {
    _outputShapeValue(index)
  }

  /**
   * There's no batch dim in the outputShape which just represent a sample record.
   */
  private[bigdl] def getOutputShape(): Shape = {
    if (_outputShapeValue.length > 1) {
      throw new RuntimeException(
        "There's multipule output for this layer. Please use getInputShapeFor instead")
    }
    outputShapeValue(0)
  }

  /**
   * Execute builing logic and return the outputShape for the given inputShape.
   * NB: the first dim of inputShape is batch
   */
  private[bigdl] def build(inputShape: Shape): Shape = {
    val outputShape = computeOutputShape(inputShape)
    this._outputShapeValue.append(outputShape)
    this._inputShapeValue = inputShape
    isBuilt = true
    outputShape
  }

  private[bigdl] var isBuilt: Boolean = false

  /**
   * There's no batch dim in the inputShape which just represent a sample record.
   */
  private[bigdl] def computeOutputShape(inputShape: Shape): Shape = {
    throw new RuntimeException("Haven't been implemented yet. Do not use it with Keras Layer")
  }
}

