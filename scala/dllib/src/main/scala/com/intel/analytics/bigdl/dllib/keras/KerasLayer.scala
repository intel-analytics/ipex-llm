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
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{Sequential => KSequential}
import com.intel.analytics.bigdl.nn.{Graph, StaticGraph, Container => TContainer, Input => TInput, Sequential => TSequential}
import com.intel.analytics.bigdl.serialization.Bigdl.{AttrValue, BigDLModule}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.serializer._
import com.intel.analytics.bigdl.utils.serializer.converters.DataConverter
import com.intel.analytics.bigdl.utils.{MultiShape, Shape, SingleShape}

import scala.collection.JavaConverters._
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

object KerasLayerSerializer extends KerasLayerSerializable

trait KerasLayerSerializable extends ContainerSerializable with TKerasSerializerHelper{

  override def loadSubModules[T: ClassTag](context : DeserializeContext,
      module : AbstractModule[Activity, Activity, T])
    (implicit ev: TensorNumeric[T]) : Unit = {
    val klayer = module.asInstanceOf[KerasLayer[Activity, Activity, T]]
    val subModules = context.bigdlModule.getSubModulesList.asScala
    subModules.foreach(module => {
      val subModuleData = ModuleSerializer.load(DeserializeContext(module,
        context.storages, context.storageType, _copyWeightAndBias))
      klayer.labor = subModuleData.module
    })
  }

  override def doSerializeModule[T: ClassTag](context: SerializeContext[T],
                                              moduleBuilder : BigDLModule.Builder)
                                             (implicit ev: TensorNumeric[T]) : Unit = {
    super.doSerializeModule(context, moduleBuilder)
    appendKerasLabel(context, moduleBuilder)
  }
}

/**
 * Wrap a torch style layer to keras style layer.
 * This layer can be built multiple times.
 * We are supposing the inputshape and the outputshape keep the same in this layer.
 * @param layer a torch style layer
 * @return a keras compatible layer
 */
class KerasIdentityWrapper[T: ClassTag]
(val layer: AbstractModule[Activity, Activity, T])(implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T](null) {
  if (layer.isKerasStyle()) {
    throw new RuntimeException(s"We only accept torch layer here, but got: $layer")
  }
  override def computeOutputShape(inputShape: Shape): Shape = {
    inputShape
  }
  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = layer
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
  extends KerasLayer[Activity, Activity, T](KerasLayer.addBatch(inputShape)) {

  require(!torchLayer.isKerasStyle(), s"We only accept torch layer here, but got: $torchLayer")

  override def computeOutputShape(calcInputShape: Shape): Shape = {
    val dummyOutTensor =
      torchLayer.cloneModule().forward(Tensor[T](
        (List(2) ++ KerasLayer.removeBatch(calcInputShape).toSingle()).toArray).fill(ev.one))
    val outSize = dummyOutTensor.toTensor.size()
    KerasLayer.addBatch(Shape(outSize.slice(1, outSize.length)))
  }

  override def doBuild(inputShape: Shape): AbstractModule[Activity, Activity, T] = torchLayer
}

private[bigdl] object KerasLayer {
  private[bigdl] def fuse[T: ClassTag](torchLayer: AbstractModule[Activity, Activity, T],
        kerasActivation: KerasLayer[Tensor[T], Tensor[T], T],
        batchInputShape: Shape)
        (implicit ev: TensorNumeric[T]): AbstractModule[Activity, Activity, T] = {
    if (kerasActivation == null) {
      torchLayer
    } else {
      val wrapper = KSequential[T]()
      wrapper.add(new KerasLayerWrapper[T](torchLayer,
        KerasLayer.removeBatch(batchInputShape)))
      wrapper.add(kerasActivation)
      wrapper.setName(torchLayer.getName())
      wrapper.build(batchInputShape)
      wrapper
    }
  }

  private[bigdl] def addBatch(shape: Shape): Shape = {
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

  private[bigdl] def removeBatch(shape: Shape): Shape = {
    // simply return null here as null is the default value
    if (shape == null) {
      return null
    }
    if (shape.isInstanceOf[SingleShape]) {
      Shape((shape.toSingle().slice(1, shape.toSingle().length)).toArray)
    } else {
      Shape(shape.toMulti().map {removeBatch(_)})
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
(batchInputShape: Shape = null)(implicit ev: TensorNumeric[T]) extends TContainer[A, B, T] {

  inputShapeValue = batchInputShape

  override def getEndNodes(startNodes: Array[ModuleNode[T]]): Array[ModuleNode[T]] = {
    if (this.isKerasGraph()) {
      this.toGraph().getEndNodes(startNodes)
    } else if (labor.isKerasStyle() && labor.getName().equals(this.getName())) {
      Array(this.processInputs(startNodes))
    } else {
      labor.getEndNodes(startNodes)
    }
  }

  override def toGraph(startNodes: ModuleNode[T]*): Graph[T] = {
    if (this.isKerasGraph()) {
      val graph = labor.asInstanceOf[StaticGraph[T]]
      val fwdExecutions = graph.getSortedForwardExecutions()
      for (i <- 0 until fwdExecutions.length) {
        val layer = fwdExecutions(i).element.asInstanceOf[KerasLayer[Activity, Activity, T]]
        if (layer.isKerasContainer()) {
          fwdExecutions(i).element = layer.toGraph()
        } else if ((!layer.labor.isKerasStyle()
          && layer.labor.isInstanceOf[TContainer[Activity, Activity, T]]) ||
          (layer.isKerasStyle() && layer.labor.isKerasStyle() &&
            layer.labor.asInstanceOf[KerasLayer[Activity, Activity, T]].isKerasContainer())) {
          fwdExecutions(i).element = layer.labor.toGraph()
        } else {
          fwdExecutions(i).element = layer.labor
        }
      }
      val result = graph.toSingleGraph()
      if (inputsFormats != null) {
        result.setInputFormats(inputsFormats)
      }

      if (inputsFormats != null) {
        result.setOutputFormats(outputsFormats)
      }
      result
    } else if (this.isKerasSequential()) {
      val starts = if (startNodes.isEmpty) Array(TInput[T]()) else startNodes.toArray
      val endNodes = this.getEndNodes(starts)
      // Disable excludeInvalidLayers to allow customized Keras layers
      val result = new StaticGraph(starts, endNodes, enableExcludeChecking = false).toSingleGraph()
      if (inputsFormats != null) {
        result.setInputFormats(inputsFormats)
      }

      if (outputsFormats != null) {
        result.setOutputFormats(outputsFormats)
      }
      result
    } else {
      this.labor.toGraph()
    }
  }

  private def isKerasGraph(): Boolean = {
    if (labor.isInstanceOf[StaticGraph[T]]) {
      val fwdExecutions = labor.asInstanceOf[StaticGraph[T]].getForwardExecutions()
      for (i <- 0 until fwdExecutions.length) {
        if (!fwdExecutions(i).element.isKerasStyle()) {
          return false
        }
      }
      true
    } else {
      false
    }
  }

  private def isKerasSequential(): Boolean = {
    if (labor.isInstanceOf[TSequential[T]]) {
      for (i <- 0 until labor.asInstanceOf[TSequential[T]].modules.length) {
        if (!labor.asInstanceOf[TSequential[T]].modules(i).isKerasStyle()) {
          return false
        }
      }
      true
    } else {
      false
    }
  }

  private def isKerasContainer(): Boolean = {
    isKerasGraph() || isKerasSequential()
  }

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

  override def isBuilt(): Boolean = {
    !this.modules.isEmpty && super.isBuilt()
  }

  override def isKerasStyle(): Boolean = true

  override def computeOutputShape(inputShape: Shape): Shape = {
    labor.computeOutputShape(inputShape)
  }

  private[bigdl] def checkWithCurrentInputShape(calcInputShape: Shape): Unit = {
    if (getInputShape() != null) {
      val withoutBatchInputShape = KerasLayer.removeBatch(getInputShape())
      val withoutBatchCalcInputShape = KerasLayer.removeBatch(calcInputShape)
      require(withoutBatchInputShape == withoutBatchCalcInputShape,
        s"InputShape from constructor ${withoutBatchInputShape}" +
          s"should be the same with the calculated inputShape: ${withoutBatchCalcInputShape}")
    }
  }

  override def build(calcInputShape: Shape): Shape = {
    // Input would be reused multiple time in inputs for StaticGraph
    if (isBuilt() && !this.allowRebuilt()) {
      throw new RuntimeException(s"Should not build this module: $this multiple times")
    }
    labor = doBuild(calcInputShape)
    checkWithCurrentInputShape(calcInputShape)
    super.build(calcInputShape)
  }

  /**
   * The value return by this method should be able to execute `forward` directly.
   */
  def doBuild(inputShape: Shape): AbstractModule[A, B, T]

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes
   * @return node containing current module
   */
  override def inputs(nodes : ModuleNode[T]*): ModuleNode[T] = {
    validateInput(nodes.map(_.element))
    if (!nodes.isEmpty) { // as there's Identity().inputs() within Graph
    val inputShape = Shape(nodes.map{_.element.getOutputShape()}.toList)
      this.build(inputShape)
    }

    processInputs(nodes)
  }

  /**
   * Build graph: some other modules point to current module
   * @param nodes upstream module nodes in an array
   * @return node containing current module
   */
  override def inputs(nodes : Array[ModuleNode[T]]): ModuleNode[T] = {
    validateInput(nodes.map(_.element))
    if (!nodes.isEmpty) {
    val inputShape = Shape(nodes.map{_.element.getOutputShape()}.toList)
      this.build(inputShape)
    }
    processInputs(nodes)
  }

  private def getShapeByIndex(shape: Shape, index: Int): Shape = {
    shape match {
      case s: SingleShape =>
        require(index == 1, s"Getting singleshape but with index: $index")
        s
      case m: MultiShape =>
        val multiShape = m.toMulti()
        require(index >= 1 && index <= multiShape.length)
        multiShape(index - 1)
    }
  }

  /**
   * Build graph: some other modules point to current module
   * @param first distinguish from another inputs when input parameter list is empty
   * @param nodesWithIndex upstream module nodes and the output tensor index. The start index is 1.
   * @return node containing current module
   */
  override def inputs(first: (ModuleNode[T], Int),
     nodesWithIndex : (ModuleNode[T], Int)*): ModuleNode[T] = {
    validateInput(List(first._1.element))
    val shapes = ArrayBuffer[Shape]()
    shapes += getShapeByIndex(first._1.element.getOutputShape(), first._2)
    if (!nodesWithIndex.isEmpty) {
      validateInput(nodesWithIndex.map(_._1.element))
      shapes ++= nodesWithIndex.map{t =>
        getShapeByIndex(first._1.element.getOutputShape(), first._2)
      }
    }
    this.build(Shape(shapes.toList))
    processInputs(first, nodesWithIndex : _*)
  }
}
