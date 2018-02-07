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

import com.intel.analytics.bigdl.utils.Shape

trait InferShape {

  private[bigdl] var _inputShapeValue: Shape = null

  private[bigdl] var _outputShapeValue: Array[Shape] = Array[Shape]()

  private[bigdl] def inputShapeValue: Shape = _inputShapeValue

  private[bigdl] def outputShapeValue: Array[Shape] = _outputShapeValue

  // scalastyle:off
  private[bigdl] def inputShapeValue_=(value: Shape): Unit = {
    _inputShapeValue = value
  }

  private[bigdl] def outputShapeValue_=(value: Array[Shape]): Unit = {
    _outputShapeValue = value
  }
  // scalastyle:on

  /**
   * We suppose the first dim is batch
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
   * We suppose the first dim is batch
   */
  private[bigdl] def getOutputShape(): Shape = {
    if (_outputShapeValue.length > 1) {
      throw new RuntimeException(
        "There are multiple outputs for this layer. Please use getInputShapeFor instead")
    }
    outputShapeValue(0)
  }

  /**
   * Execute building logic and return the outputShape for the given inputShape.
   * NB: the first dim of inputShape is batch
   */
  private[bigdl] def build(inputShape: Shape): Shape = {
    val outputShape = computeOutputShape(inputShape)
    this._outputShapeValue ++ Array(outputShape)
    this._inputShapeValue = inputShape
    isBuilt = true
    outputShape
  }

  private[bigdl] var isBuilt: Boolean = false


  private[bigdl] def isCompatibleWithKeras(): Boolean = true

  private[bigdl] def isCompatibleWithTorch(): Boolean = true

  /**
   * We suppose the first dim is batch
   */
  private[bigdl] def computeOutputShape(inputShape: Shape): Shape = {
    throw new RuntimeException("Haven't been implemented yet. Do not use it with Keras Layer")
  }
}

