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

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity, TensorModule}
import com.intel.analytics.bigdl.nn.keras.{Sequential => KSequential}
import com.intel.analytics.bigdl.nn.{Container, Sequential => TSequential}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{Shape, SingleShape, Util}
import serialization.Bigdl.{AttrValue, BigDLModule}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

private[bigdl] trait TKerasSerializerHelper {
  def appendKerasLabel[T: ClassTag](context: SerializeContext[T],
                       moduleBuilder : BigDLModule.Builder)(implicit ev: TensorNumeric[T]): Unit = {
    val serializerFlagBuilder = AttrValue.newBuilder
    DataConverter.setAttributeValue(context, serializerFlagBuilder, true,
      scala.reflect.runtime.universe.typeOf[Boolean])
    moduleBuilder.putAttr("is_keras_module", serializerFlagBuilder.build)
  }
}

object KerasLayerSerializer extends ContainerSerializable with TKerasSerializerHelper{

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              moduleBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {
    super.doSerializeModule(context, moduleBuilder)
    appendKerasLabel(context, moduleBuilder)
  }
}

private[bigdl] object KerasLayer {
    def fuse[T: ClassTag](sLayer: AbstractModule[Activity, Activity, T],
        activation: AbstractModule[Tensor[T], Tensor[T], T],
        inputShape: Shape)
        (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
      if (activation == null) {
        return sLayer
      }
      val seq = KSequential[T]()
      seq.add(InputLayer(inputShape = KerasLayer.removeBatch(inputShape)))
      seq.add(sLayer)
      seq.add(activation)
      seq.setName(sLayer.getName())
      seq
    }

   def addBatch(shape: Shape): Shape = {
     // simply return null here as null is the default value
     if (shape == null) {
      return null
    }
    if (shape.isInstanceOf[SingleShape]) {
      Shape((List(-1) ++ shape.toSingle()).toArray)
    } else {
      Shape(shape.toMulti().map {addBatch(_)})
    }
  }

  def removeBatch(shape: Shape): Shape = {
    // simply return null here as null is the default value
    if (shape == null) {
      return null
    }
    if (shape.isInstanceOf[SingleShape]) {
      Shape((shape.toSingle().slice(1, shape.toSingle().length)).toArray)
    } else {
      Shape(shape.toMulti().map {addBatch(_)})
    }
  }
}

/**
 * KerasModule is the basic component of all Keras-like Layer.
 * It forward activities and backward gradients, and can be mixed with other AbstractMoudule.
 *
 * @tparam A Input data type
 * @tparam B Output data type
 * @tparam T Numeric type of parameter(e.g. weight, bias). Only support float/double now
 * @param batchInputShape the first dim is batch
 */
abstract class KerasLayer[A <: Activity: ClassTag, B <: Activity: ClassTag, T: ClassTag]
(batchInputShape: Shape = null)(implicit ev: TensorNumeric[T]) extends Container[A, B, T]{

  def labor: AbstractModule[A, B, T] = {
    if (this.modules.isEmpty) {
      throw new RuntimeException("This Layer hasn't been built")
    }
    require(modules.length == 1,
      s"modules should only contain 1 element instead of ${modules.length}")
    modules(0).asInstanceOf[AbstractModule[A, B, T]]
  }

  // scalastyle:off
  def labor_=(value: AbstractModule[A, B, T]): Unit = {
    modules.clear()
    modules.append(value)
  }
  // scalastyle:on
  override def inputShapeValue: Shape = labor.inputShapeValue

  override def outputShapeValue: Array[Shape] = labor.outputShapeValue

  // scalastyle:off
  override def inputShapeValue_=(value: Shape): Unit = {
    labor.inputShapeValue = value
    this._inputShapeValue = value
  }

  override def outputShapeValue_=(value: Array[Shape]): Unit = {
   labor.outputShapeValue = value
   this._outputShapeValue = value
  }
  // scalastyle:on

  override def updateOutput(input: A): B = {
    output = labor.updateOutput(input)
    output
  }

  override def updateGradInput(input: A, gradOutput: B): A = {
    gradInput = labor.updateGradInput(input, gradOutput)
    gradInput
  }

  override def accGradParameters(input: A, gradOutput: B): Unit = {
    labor.accGradParameters(input, gradOutput)
  }

  override def isCompatibleWithKeras(): Boolean = true

  override def isCompatibleWithTorch(): Boolean = false

  override def getInputShape(): Shape = {
    if (batchInputShape != null) {
      batchInputShape
    } else if (this.labor == null) {
      null
    } else {
      this.labor.getInputShape()
    }
  }

  override def computeOutputShape(inputShape: Shape): Shape = {
    this.labor.computeOutputShape(inputShape)
  }

  override def getOutputShape(): Shape = labor.getOutputShape()

  override def build(inputShape: Shape): Shape = {
    this.labor = doBuild(inputShape)
    val outputShape = computeOutputShape(inputShape)
    this.outputShapeValue ++= Array(outputShape)
    this.inputShapeValue = inputShape
    isBuilt = true
    outputShape // we cannot use getOutputShape here as it may containing multiple value
  }

  def doBuild(inputShape: Shape): AbstractModule[A, B, T]

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes
   * @return node containing current module
   */
  override def inputs(nodes : ModuleNode[T]*): ModuleNode[T] = {
    Util.excludeNotKeras(nodes.map(_.element))
    if (!nodes.isEmpty) { // as there's  Identity().inputs() within Graph
    val inputShape = Shape(nodes.map{_.element.getOutputShape()}.toList)
      this.build(inputShape)
    }
    super.inputs(nodes: _*)
  }

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes in an array
   * @return node containing current module
   */
  override def inputs(nodes : Array[ModuleNode[T]]): ModuleNode[T] = {
    Util.excludeNotKeras(nodes.map(_.element))
    if (!nodes.isEmpty) { // as there's  Identity().inputs() within Graph
    val inputShape = Shape(nodes.map{_.element.getOutputShape()}.toList)
      this.build(inputShape)
    }
    super.inputs(nodes)
  }

  /**
   * Build graph: some other modules point to current module
   * @param first distinguish from another inputs when input parameter list is empty
   * @param nodesWithIndex upstream module nodes and the output tensor index. The start index is 1.
   * @return node containing current module
   */
  override def inputs(first: (ModuleNode[T], Int),
     nodesWithIndex : (ModuleNode[T], Int)*): ModuleNode[T] = {
    Util.excludeNotKeras(List(first._1.element))
    Util.excludeNotKeras(nodesWithIndex.map(_._1.element))
    val shapes = ArrayBuffer[Shape]()
    shapes.append(first._1.element.getOutputShapeFor(first._2))
    if (!nodesWithIndex.isEmpty) {
      shapes ++= nodesWithIndex.map{t => t._1.element.getOutputShapeFor(t._2)}
    }
    this.build(Shape(shapes.toList))
    super.inputs(first, nodesWithIndex : _*)
  }
}
