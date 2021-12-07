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

package com.intel.analytics.bigdl.dllib.keras

import java.io.{File, FilenameFilter}
import java.text.SimpleDateFormat
import java.util.Calendar

import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.mkl.MKL
import com.intel.analytics.bigdl.dllib.feature.dataset.{MiniBatch, _}
import com.intel.analytics.bigdl.dllib.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.dllib.feature.dataset.DataSet
import com.intel.analytics.bigdl.dllib.optim
import com.intel.analytics.bigdl.dllib._
import com.intel.analytics.bigdl.dllib.nn.Graph._
import com.intel.analytics.bigdl.dllib.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.dllib.nn.internal.{KerasLayer, KerasLayerSerializable}
import com.intel.analytics.bigdl.dllib.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.dllib.nn.{Container, Graph, Module, StaticGraph, Sequential => TSequential}
import com.intel.analytics.bigdl.dllib.optim.DistriOptimizer.{Cache, CacheV1}
import com.intel.analytics.bigdl.dllib.optim.DistriOptimizerV2.{Cache => CacheV2}
import com.intel.analytics.bigdl.dllib.optim._
import com.intel.analytics.bigdl.dllib.optim.parameters.AllReduceParameter
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.utils._
import com.intel.analytics.bigdl.dllib.utils.serializer.{DeserializeContext, ModuleData, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.dllib.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.dllib.optim.ZooTrigger
import com.intel.analytics.bigdl.dllib.feature.{DiskFeatureSet, DistributedFeatureSet, FeatureSet}
import com.intel.analytics.bigdl.dllib.feature.image.ImageSet
import com.intel.analytics.bigdl.dllib.feature.dataset
import com.intel.analytics.bigdl.dllib.feature.text._
import com.intel.analytics.bigdl.dllib.keras.{Net, Predictable}
import com.intel.analytics.bigdl.dllib.keras.autograd.{Lambda, Variable}
import com.intel.analytics.bigdl.dllib.keras.autograd._
import com.intel.analytics.bigdl.dllib.keras.layers.Input
import com.intel.analytics.bigdl.dllib.keras.layers.utils._
import com.intel.analytics.bigdl.dllib.keras.models._
import com.intel.analytics.bigdl.dllib.net.NetUtils
// import com.intel.analytics.bigdl.dllib.Net.TorchModel
import com.intel.analytics.bigdl.dllib.estimator.{AbstractEstimator, ConstantClipping, GradientClipping, L2NormClipping}
// import com.intel.analytics.zoo.tfpark.{TFTrainingHelper, TFTrainingHelperV2}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.commons.lang3.SerializationUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.log4j.Logger
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.language.implicitConversions

class Model[T: ClassTag] private (private val _inputs : Seq[ModuleNode[T]],
  private val _outputs : Seq[ModuleNode[T]])(implicit ev: TensorNumeric[T])
  extends KerasNet[T] with NetUtils[T, Model[T]] {
  this.labor = doBuild(null)
  KerasLayerRef(this).excludeInvalidLayers(this.labor.asInstanceOf[StaticGraph[T]].
    getForwardExecutions().map {_.element})

  KerasLayerRef(this).setInputShape(Shape(_inputs.map{n => n.element.getInputShape()}.toList))

  KerasLayerRef(this).setOutShape(Shape(_outputs.map{_.element.getOutputShape()}.toList))

  private[bigdl] def getInputs(): Seq[ModuleNode[T]] = _inputs

  private[bigdl] def getOutputs(): Seq[ModuleNode[T]] = _outputs

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
   * Save the current model graph to a folder, which can be displayed in TensorBoard
   * by running the command:
   * tensorboard --logdir logPath
   *
   * @param logPath The path to save the model graph.
   * @param backward Whether to draw backward graph instead of forward.
   * @return
   */
  def saveGraphTopology(logPath: String, backward: Boolean = false): this.type = {
    this.labor.asInstanceOf[Graph[T]].saveGraphTopology(logPath, backward)
    this
  }

  override def unFreeze(names: String*): Model.this.type = {
    labor.unFreeze(names: _*)
    this
  }

  private val graph = labor.asInstanceOf[Graph[T]]

  override def nodes(names: Seq[String]): Seq[ModuleNode[T]] = {
    names.map(graph.node)
  }

  override def node(name: String): ModuleNode[T] = {
    graph.node(name)
  }

  override def newGraph(output: String): Model[T] = {
    new Model[T](_inputs, nodes(Seq(output)).map(_.removeNextEdges()))
  }

  override def newGraph(outputs: Seq[String]): Model[T] = {
    new Model[T](_inputs, nodes(outputs).map(_.removeNextEdges()))
  }

  override def toModel(): Model[T] = this

  override def toKeras(): Model[T] = this

  override private[bigdl] def getKerasWeights(): Array[Tensor[Float]] = {
    val weights = new ArrayBuffer[Tensor[Float]]()
    modules(0).asInstanceOf[StaticGraph[T]].modules.foreach(m => {
      val params = m.asInstanceOf[Net].getKerasWeights()
      if (params != null) {
        params.foreach(weights += _)
      }
    })
    weights.toArray
  }


  override def summary(
                        lineLength: Int = 120,
                        positions: Array[Double] = Array(.33, .55, .67, 1)): Unit = {
    println("Model Summary:")
    KerasUtils.printSplitLine('-', lineLength)
    val toDisplay = Array("Layer (type)", "Output Shape", "Param #", "Connected to")
    KerasUtils.printRow(toDisplay, lineLength, positions, splitChar = '=')
    val nodes = labor.asInstanceOf[StaticGraph[T]].getSortedForwardExecutions()
    var totalParams = 0
    var trainableParams = 0
    for (node <- nodes) {
      val (total, trainable) = KerasUtils.printNodeSummary(node, lineLength, positions)
      totalParams += total
      trainableParams += trainable
    }
    println("Total params: " + "%,d".format(totalParams))
    println("Trainable params: " + "%,d".format(trainableParams))
    println("Non-trainable params: " + "%,d".format(totalParams - trainableParams))
    KerasUtils.printSplitLine('-', lineLength)
  }
}

object Model extends KerasLayerSerializable {
  ModuleSerializer.registerModule(
    "com.intel.analytics.bigdl.dllib.keras.Model",
    Model)

  /**
   * Build a multiple-input, multiple-output graph container.
   * @param input Array of input nodes.
   * @param output Array of output nodes.
   * @return A graph container.
   */
  def apply[T: ClassTag](
    input : Array[ModuleNode[T]],
    output : Array[ModuleNode[T]])(implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](input, output)
  }

  /**
   * Build a single-input, multiple-output graph container
   * @param input The input node.
   * @param output Array of output nodes.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : Array[ModuleNode[T]])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input), output)
  }

  /**
   * Build a multiple-input, single-output graph container.
   * @param input Array of input nodes.
   * @param output The output node.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : Array[ModuleNode[T]], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](input, Seq(output))
  }

  /**
   * Build a single-input, single-output graph container
   * @param input The input node.
   * @param output The output node.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input), Seq(output))
  }

  /* ------------------------ factory methods for variables--------------------- */
  /**
   * Build a multiple-input, multiple-output graph container.
   * @param input Array of input variables.
   * @param output Array of output variables.
   * @return A graph container.
   */
  def apply[T: ClassTag](
    input : Array[Variable[T]],
    output : Array[Variable[T]])(implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](input.map(_.node), output.map(_.node))
  }

  /**
   * Build a single-input, multiple-output graph container
   * @param input The input variable.
   * @param output Array of output variables.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : Variable[T], output : Array[Variable[T]])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input.node), output.map(_.node))
  }

  /**
   * Build a multiple-input, single-output graph container.
   * @param input Array of input variables.
   * @param output The output variables.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : Array[Variable[T]], output : Variable[T])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](input.map(_.node), Seq(output.node))
  }

  /**
   * Build a single-input, single-output graph container
   * @param input The input variable.
   * @param output The output variable.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : Variable[T], output : Variable[T])
    (implicit ev: TensorNumeric[T]) : Model[T] = {
    new Model[T](Seq(input.node), Seq(output.node))
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
