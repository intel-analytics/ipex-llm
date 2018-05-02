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

package com.intel.analytics.zoo.pipeline.api.keras.models

import com.intel.analytics.bigdl.nn.Graph.ModuleNode
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasLayerSerializable, KerasModel, Model => BModel, Sequential => BSequential}
import com.intel.analytics.bigdl.nn.{Graph, StaticGraph, Sequential => TSequential}
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleData, ModuleSerializer, SerializeContext}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.{AbstractModuleRef, GraphRef, KerasLayerRef}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag


abstract class KerasNet[T: ClassTag](
    implicit ev: TensorNumeric[T]) extends KerasModel[T] {
}

class Model[T: ClassTag] private (private val _inputs : Seq[ModuleNode[T]],
    private val _outputs : Seq[ModuleNode[T]])(implicit ev: TensorNumeric[T])
  extends KerasNet[T] {
  this.labor = doBuild(null)
  KerasLayerRef(this).excludeInvalidLayers(this.labor.asInstanceOf[StaticGraph[T]].
    getForwardExecutions().map {_.element})

  KerasLayerRef(this).setInputShape(Shape(_inputs.map{n => n.element.getInputShape()}.toList))

  KerasLayerRef(this).setOutShape(Shape(_outputs.map{_.element.getOutputShape()}.toList))

  override def isKerasStyle(): Boolean = true

  override def computeOutputShape(inputShape: Shape): Shape = {
    getOutputShape()
  }

  override def doBuild(inputShape: Shape): StaticGraph[T] =
    new StaticGraph[T](_inputs, _outputs, None, false)

  override def build(calcInputShape: Shape): Shape = {
    KerasLayerRef(this).checkWithCurrentInputShape(calcInputShape)
    getOutputShape()
  }

  /**
   * Save current model graph to a folder, which can be display in tensorboard by running
   *   tensorboard --logdir logPath
   * @param logPath
   * @param backward Draw backward graph instead of forward
   * @return
   */
  def saveGraphTopology(logPath: String, backward: Boolean = false): this.type = {
    this.labor.asInstanceOf[Graph[T]].saveGraphTopology(logPath, backward)
    this
  }
}

object Model extends KerasLayerSerializable{
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.pipeline.api.keras.models.Model",
    Model)

  /**
   * Build multiple inputs, multiple outputs graph container.
   * @param input input node
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](
      input : Array[ModuleNode[T]],
      output : Array[ModuleNode[T]])(implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](input, output)
  }

  /**
   * Build a single input, multiple outputs graph container
   * @param input input node
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : Array[ModuleNode[T]])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input), output)
  }

  /**
   * Build a multiple inputs, single output graph container
   * @param input input nodes
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](input : Array[ModuleNode[T]], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](input, Seq(output))
  }
  /**
   * Build a single input, single output graph container
   * @param input input nodes
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input), Seq(output))
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
      builder: BigDLModule.Builder)
    (implicit ev: TensorNumeric[T]): Unit = {
    val labor = context.moduleData.module.
      asInstanceOf[KerasLayer[Activity, Activity, T]].labor
    val subModule = ModuleSerializer.serialize(SerializeContext(ModuleData(labor,
      new ArrayBuffer[String](), new ArrayBuffer[String]()), context.storages,
      context.storageType, _copyWeightAndBias))
    builder.addSubModules(subModule.bigDLModule)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    val subProtoModules = context.bigdlModule.getSubModulesList.asScala
    val subModules = subProtoModules.map(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
        context.storages, context.storageType, _copyWeightAndBias))
      subModuleData.module
    })
    val tGraph = subModules(0).asInstanceOf[StaticGraph[T]]
    Model(tGraph.inputs.toArray, new GraphRef(tGraph).getOutputs().toArray)
  }

}

class Sequential[T: ClassTag] private ()
  (implicit ev: TensorNumeric[T]) extends KerasNet[T] {

  private[zoo] var frozen: Boolean = false

  this.labor = doBuild(null)

  private def triggerBuilding(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Unit = {
    val absModuleRef =
      new AbstractModuleRef(module.asInstanceOf[AbstractModule[Activity, Activity, T]])
    val kerasLayerRef = KerasLayerRef(this)

    if (!this.isBuilt()) {
      if (module.getInputShape() == null) {
        throw new RuntimeException("The first layer should explicitly declare inputshape")
      } else {

        val outputShape = absModuleRef.build(module.getInputShape())
        // The inputShape of Sequential should only be init here.
        kerasLayerRef.setInputShape(module.getInputShape())
        kerasLayerRef.setOutShape(outputShape)
      }
    } else {
      val outputShape = absModuleRef.build(this.getOutputShape())
      kerasLayerRef.setOutShape(outputShape)
    }
  }

  /**
   * Add a sub-module to the contained `modules`
   *
   * @param module module to be add
   * @return this container
   */
  def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    if (frozen) {
      throw new RuntimeException(
        "This Sequential has been frozen, as it has been added into other container")
    }
    if (module.isInstanceOf[Sequential[T]]) {
      module.asInstanceOf[Sequential[T]].frozen = true
    }
    val kerasLayerRef = KerasLayerRef(this)
    kerasLayerRef.validateInput[T](Seq(module))

    triggerBuilding(module)

    labor.asInstanceOf[TSequential[T]].modules +=
      module.asInstanceOf[AbstractModule[Activity, Activity, T]]
    kerasLayerRef.checkDuplicate()
    this
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    if (labor.asInstanceOf[TSequential[T]].modules.isEmpty) {
      inputShape
    } else {
      labor.asInstanceOf[TSequential[T]].modules.last.getOutputShape()
    }
  }

  override def doBuild(inputShape: Shape): TSequential[T] = TSequential[T]()

  override def build(calcInputShape: Shape): Shape = {
    val kerasLayerRef = KerasLayerRef(this)
    kerasLayerRef.checkWithCurrentInputShape(calcInputShape)
    getOutputShape()
  }
}

object Sequential extends KerasLayerSerializable{
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.pipeline.api.keras.models.Sequential",
    Sequential)

  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }
}

