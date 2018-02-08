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
import com.intel.analytics.bigdl.nn.{Graph, GraphSerializable, StaticGraph, Sequential => TSequential}
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.{Shape, Util}

import scala.reflect.ClassTag

class Model[T: ClassTag](private val _inputs : Seq[ModuleNode[T]],
      private val _outputs : Seq[ModuleNode[T]])(implicit ev: TensorNumeric[T])
  extends StaticGraph[T](_inputs, _outputs, None, false) {

  Util.excludeNotKeras(inputs.map(_.element))
  Util.excludeNotKeras(outputs.map(_.element))

  this.inputShapeValue = Shape(inputs.map{n => n.element.getInputShape()}.toList)

  this.outputShapeValue = Array(outputs.map{_.element.getOutputShape()}: _*)

  isBuilt = true

  override private[bigdl] def isCompatibleWithKeras(): Boolean = true

  override private[bigdl] def isCompatibleWithTorch(): Boolean = false

  override def computeOutputShape(inputShape: Shape): Shape = {
    getOutputShape()
  }
}

object Model extends ModelSerializer{
  /**
   * Build multiple inputs, multiple outputs graph container.
   * @param input input node
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](
      input : Array[ModuleNode[T]],
      output : Array[ModuleNode[T]])(implicit ev: TensorNumeric[T]) : Graph[T] = {
    new Model[T](input, output)
  }

  /**
   * Build a single input, multiple outputs graph container
   * @param input input node
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : Array[ModuleNode[T]])
                        (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new Model[T](Seq(input), output)
  }

  /**
   * Build a multiple inputs, single output graph container
   * @param input input nodes
   * @param output output node
   * @return a graph container
   */
  def apply[T: ClassTag](input : Array[ModuleNode[T]], output : ModuleNode[T])
                        (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new Model[T](input, Seq(output))
  }
  /**
   * Build a single input, single output graph container
   * @param input input nodes
   * @param output output nodes
   * @return a graph container
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : ModuleNode[T])
                        (implicit ev: TensorNumeric[T]) : Graph[T] = {
    new Model[T](Seq(input), Seq(output))
  }
}

trait ModelSerializer extends GraphSerializable with TKerasSerializerHelper{

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              moduleBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {
    super.doSerializeModule(context, moduleBuilder)
    appendKerasLabel(context, moduleBuilder)
  }

  override def doLoadModule[T: ClassTag](context: DeserializeContext)
    (implicit ev: TensorNumeric[T]) : AbstractModule[Activity, Activity, T] = {
    val (module, inputs, outputs, generateBackwardValue, sharedVariables) =
      prepareLoadModule(context)
    require(generateBackwardValue == null, "there's no generateBackward for keras module")
    require(module.containsAttr("is_keras_module")
      && module.getAttrOrThrow("is_keras_module").getBoolValue(), "It should be a keras module")
    Model(inputs.toArray, outputs.toArray)
  }
}

class Sequential[T: ClassTag](val stopInferShape: Boolean = false)
(implicit ev: TensorNumeric[T]) extends TSequential[T] {

  override private[bigdl] def isCompatibleWithKeras(): Boolean = true

  override private[bigdl] def isCompatibleWithTorch(): Boolean = false

  private[bigdl] var frozen: Boolean = false

  override def computeOutputShape(inputShape: Shape): Shape = {
     getOutputShape()
  }

  override def getOutputShape(): Shape = {
    require(outputShapeValue.length > 0, "Sequence should not be empty")
    outputShapeValue(outputShapeValue.length -1) // For Seq, we only respect the last item as output
  }

  private def triggerBuilding(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Unit = {
    if (this.modules.isEmpty) {
      if (module.getInputShape() == null) {
        throw new RuntimeException("The first layer should explicitly declare inputshape")
      } else {
        val outputShape = module.build(module.getInputShape())
        this.inputShapeValue = module.getInputShape()
        this.outputShapeValue = Array(outputShape)
      }
    } else {
      val outputShape = module.build(this.getOutputShape())
      this.outputShapeValue = Array(outputShape)
    }
    isBuilt = true
  }

  /**
   * Add a sub-module to the contained `modules`
   *
   * @param module module to be add
   * @return this container
   */
  override def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    if (frozen) {
      throw new RuntimeException(
        "This Sequential has been frozen, as it has been added into other container")
    }
    if (module.isInstanceOf[Sequential[T]]) {
      module.asInstanceOf[Sequential[T]].frozen = true
    }
    Util.excludeNotKeras[T](Seq(module))
    if (!stopInferShape) {
      triggerBuilding(module)
    }
    modules += module.asInstanceOf[AbstractModule[Activity, Activity, T]]
    this
  }
}

object Sequential extends ContainerSerializable with TKerasSerializerHelper{
  def apply[@specialized(Float, Double) T: ClassTag]()
     (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              moduleBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {
    super.doSerializeModule(context, moduleBuilder)
    appendKerasLabel(context, moduleBuilder)
  }
}
