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

package com.intel.analytics.bigdl.dllib.keras.layers.utils

import com.intel.analytics.bigdl.dllib.nn.{Graph, MklInt8Convertible}
import com.intel.analytics.bigdl.dllib.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.KerasLayer
import com.intel.analytics.bigdl.dllib.optim.SGD
import com.intel.analytics.bigdl.dllib.utils._
import com.intel.analytics.bigdl.dllib.keras.optimizers.{Adam, AdamWeightDecay}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

object KerasLayerRef {
  def apply[T: ClassTag](instance: KerasLayer[_, _, T]): KerasLayerRef[T] = {
    new KerasLayerRef(instance)
  }
}

class KerasLayerRef[T: ClassTag](instance: KerasLayer[_, _, T]) {

  def excludeInvalidLayers[T: ClassTag]
  (modules : Seq[AbstractModule[_, _, T]]): Unit = {
    instance.excludeInvalidLayers(modules)
  }

  def setInputShape(value: Shape): Unit = {
    instance._inputShapeValue = value
  }

  def setOutShape(value: Shape): Unit = {
    instance._outputShapeValue = value
  }

  def checkWithCurrentInputShape(calcInputShape: Shape): Unit = {
    instance.checkWithCurrentInputShape(calcInputShape)
  }

  def validateInput[T: ClassTag](modules : Seq[AbstractModule[_, _, T]]): Unit = {
    instance.validateInput(modules)
  }

  def checkDuplicate(
      record: mutable.HashSet[Int] = mutable.HashSet()
  ): Unit = {
    instance.checkDuplicate(record)
  }
}

class AbstractModuleRef[T: ClassTag](instance: AbstractModule[Activity, Activity, T]) {

  def build(inputShape: Shape): Shape = {
    instance.build(inputShape)
  }
}

class GraphRef[T: ClassTag](instance: Graph[T]) {
  def getOutputs(): Seq[ModuleNode[T]] = {
    instance.outputs
  }
}

object EngineRef {
  def getCoreNumber(): Int = {
    Engine.coreNumber()
  }

  def getNodeNumber(): Int = {
    Engine.nodeNumber()
  }

  def getDefaultThreadPool(): ThreadPool = {
    Engine.default
  }

  def getEngineType(): EngineType = {
    Engine.getEngineType()
  }

  def getOptimizerVersion(): OptimizerVersion = {
    Engine.getOptimizerVersion()
  }

  def setOptimizerVersion(optimizerVersion : OptimizerVersion): Unit = {
    Engine.setOptimizerVersion(optimizerVersion)
  }

  def setCoreNumber(num: Int): Unit = {
    Engine.setCoreNumber(num)
  }
}

object SGDRef {
  def getstate[T: ClassTag](instance: Adam[T]): Table = {
    instance.state
  }

  def getstate[T: ClassTag](instance: AdamWeightDecay[T]): Table = {
    instance.state
  }

  def getstate[T](instance: SGD[T]): Table = {
    instance.state
  }
}
