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

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.KerasLayer
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Node, Shape}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.bigdl.nn.{Input => TInput}
import com.intel.analytics.zoo.pipeline.api.Net

import scala.reflect.ClassTag

class Input[T: ClassTag](val inputShape: Shape)(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](KerasUtils.addBatch(inputShape)) with Net {

  private var skipDuplicate = false

  private[Input] def setSkipDuplicate(): this.type = {
    this.skipDuplicate = true
    this
  }

  override def computeOutputShape(inputShape: Shape): Shape = inputShape

  override def doBuild(inputShape: Shape): TInput[T] = new TInput[T]()

  override def allowRebuilt(): Boolean = true

  override def skipDuplicateCheck(): Boolean = skipDuplicate

  override private[zoo] def toKeras2(): String = {
    val params = Net.inputShapeToString(inputShape, "shape") ++
      Net.param(getName())
    Net.kerasDef(this, params)
  }
}

/**
 * Used to instantiate an input node.
 */
object Input {
  def apply[T: ClassTag](
      inputShape: Shape = null,
      name : String = null)(implicit ev: TensorNumeric[T]): ModuleNode[T] = {
    // As this method will return a node, it cannot be added to a container or connect with other
    // nodes multiple times. So we can skip the duplicate checking.
    // Even it appears multiple times in a nested container, it's okay as it will always
    // be the first layer.
    val module = new Input(inputShape).setSkipDuplicate()
    module.build(KerasUtils.addBatch(inputShape))
    if (name != null) {
      module.setName(name)
    }
    new Node(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
  }
}

/**
 * Used as an entry point into a model.
 */
object InputLayer {
  def apply[T: ClassTag](
      inputShape: Shape = null,
      name : String = null)(implicit ev: TensorNumeric[T]): KerasLayer[Activity, Activity, T] = {
    val module = new Input(inputShape)
    if (name != null) {
      module.setName(name)
    }
    module
  }
}
