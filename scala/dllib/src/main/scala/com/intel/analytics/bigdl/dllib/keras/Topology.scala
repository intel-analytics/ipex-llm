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

import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Container, Identity, StaticGraph, Sequential => TSequential}
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.Shape
import com.intel.analytics.bigdl.utils.serializer._

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

abstract class KerasModel[T: ClassTag](implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T] {
  // TODO: enrich fit, compile, evaluate etc here.

  def getSubModules(): List[AbstractModule[Activity, Activity, T]] = {
    require(this.labor.isInstanceOf[Container[Activity, Activity, T]],
      "labor should be a container, but we got: $this")
    this.labor.asInstanceOf[Container[Activity, Activity, T]].modules.toList
  }
}

class Model[T: ClassTag](private val _inputs : Seq[ModuleNode[T]],
      private val _outputs : Seq[ModuleNode[T]])(implicit ev: TensorNumeric[T])
  extends KerasModel[T] {
  this.labor = doBuild(null)

  excludeInvalidLayers(this.labor.asInstanceOf[StaticGraph[T]].
    getForwardExecutions().map {_.element})

  this.inputShapeValue = Shape(_inputs.map{n => n.element.getInputShape()}.toList)

  this.outputShapeValue = Shape(_outputs.map{_.element.getOutputShape()}.toList)

  override def isKerasStyle(): Boolean = true

  override def computeOutputShape(inputShape: Shape): Shape = {
    getOutputShape()
  }

  override def doBuild(inputShape: Shape): StaticGraph[T] =
    new StaticGraph[T](_inputs, _outputs, None, false)

  override def build(calcInputShape: Shape): Shape = {
    checkWithCurrentInputShape(calcInputShape)
    getOutputShape()
  }
}

object Model extends KerasLayerSerializable{
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
    Model(tGraph.inputs.toArray, tGraph.outputs.toArray)
  }

}

class Sequential[T: ClassTag]()
(implicit ev: TensorNumeric[T]) extends KerasModel[T] {

  private[bigdl] var frozen: Boolean = false

  this.labor = doBuild(null)

  private def triggerBuilding(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Unit = {
    if (this.getOutputShape() == null) {
      if (module.getInputShape() == null) {
        throw new RuntimeException("The first layer should explicitly declare inputshape")
      } else {
        val outputShape = module.build(module.getInputShape())
        // The inputShape of Sequential should only be init here.
        this.inputShapeValue = module.getInputShape()
        this.outputShapeValue = outputShape
      }
    } else {
      val outputShape = module.build(this.getOutputShape())
      this.outputShapeValue = outputShape
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
    validateInput[T](Seq(module))

    triggerBuilding(module)

    labor.asInstanceOf[TSequential[T]].modules +=
      module.asInstanceOf[AbstractModule[Activity, Activity, T]]
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
    checkWithCurrentInputShape(calcInputShape)
    getOutputShape()
  }
}

object Sequential extends KerasLayerSerializable{
  def apply[@specialized(Float, Double) T: ClassTag]()
     (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }
}
