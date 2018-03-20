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

import com.intel.analytics.bigdl.nn.keras.{Input => KInput, Sequential => KSequential}
import com.intel.analytics.bigdl.nn.{Input => TInput}
import com.intel.analytics.bigdl.utils.Shape

import scala.language.existentials
import scala.reflect.ClassTag

class InvalidLayer(msg: String) extends RuntimeException(msg)

trait InferShape {
  private[bigdl] var _inputShapeValue: Shape = null

  private[bigdl] var _outputShapeValue: Shape = null

  private[bigdl] def inputShapeValue: Shape = _inputShapeValue

  private[bigdl] def outputShapeValue: Shape = _outputShapeValue

  // scalastyle:off
  private[bigdl] def inputShapeValue_=(value: Shape): Unit = {
    _inputShapeValue = value
  }

  private[bigdl] def outputShapeValue_=(value: Shape): Unit = {
    _outputShapeValue = value
  }
  // scalastyle:on

  /**
   * Return the inputShape for the current Layer and the first dim is batch.
   */
  final def getInputShape(): Shape = {
    require(this.isKerasStyle(),
      "Torch style definition doesn't support getInputShape for now.")
    _inputShapeValue
  }

  /**
   * Return the outputShape for the current Layer and the first dim is batch.
   */
  final def getOutputShape(): Shape = {
    require(this.isKerasStyle(),
      "Torch style definition doesn't support getOutputShape for now.")
    require(this.isBuilt(), "This module hasn't been built.")
    outputShapeValue
  }

  /**
   * Execute building logic and return the outputShape for the given inputShape.
   * NB: the first dim of inputShape is batch
   */
  private[bigdl] def build(inputShape: Shape): Shape = {
    val outputShape = computeOutputShape(inputShape)
    this.outputShapeValue = outputShape
    this.inputShapeValue = inputShape
    outputShape
  }

  private[bigdl] def isBuilt(): Boolean = outputShapeValue != null

  private[bigdl] def isKerasStyle(): Boolean = false

  private[bigdl] def allowRebuilt(): Boolean = false

  /**
   * We suppose the first dim is batch
   */
  private[bigdl] def computeOutputShape(inputShape: Shape): Shape = {
    throw new RuntimeException("Haven't been implemented yet. Do not use it with Keras Layer")
  }

  private[bigdl] def excludeInvalidLayers[T: ClassTag]
  (modules : Seq[AbstractModule[_, _, T]]): Unit = {
    val invalidNodes = if (this.isKerasStyle()) {
      modules.filter{!_.isKerasStyle()}
    } else {
      modules.filter{_.isKerasStyle()}
    }
    if (invalidNodes.length > 0) {
      throw new InvalidLayer(s"""Do not mix ${this}(isKerasStyle=${isKerasStyle()}) with Layer
                           (isKerasStyle=${invalidNodes(0).isKerasStyle()}):
         ${invalidNodes.mkString(",")}""")
    }
  }

  private[bigdl] def validateInput[T: ClassTag](modules : Seq[AbstractModule[_, _, T]]): Unit = {
    if (this.isKerasStyle()) {
      require(modules != null && !modules.isEmpty, "Empty input is not allowed")
    }
    excludeInvalidLayers(modules)
  }
}

