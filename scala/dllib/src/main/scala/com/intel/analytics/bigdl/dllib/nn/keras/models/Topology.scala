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

import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.{Criterion, DataSet}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasLayerSerializable}
import com.intel.analytics.bigdl.nn.{Container, Graph, StaticGraph, Sequential => TSequential}
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils.{Edge, LoggerFilter, Node, Shape}
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleData, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.autograd.{CustomLossWithVariable, Lambda, Variable}
import com.intel.analytics.zoo.pipeline.api.autograd._
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.{AbstractModuleRef, GraphRef, KerasLayerRef}
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.KerasUtils
import com.intel.analytics.zoo.pipeline.api.net.NetUtils
import org.apache.spark.rdd.RDD

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.language.implicitConversions



abstract class KerasNet[T: ClassTag](implicit ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T] with Net {

  def getSubModules(): List[AbstractModule[Activity, Activity, T]] = {
    require(this.labor.isInstanceOf[Container[Activity, Activity, T]],
      s"labor should be a container, but we got: $this")
    this.labor.asInstanceOf[Container[Activity, Activity, T]].modules.toList
  }

  private var optimMethod: OptimMethod[T] = null
  private var criterion: Criterion[T] = null
  private var vMethods: Array[ValidationMethod[T]] = null
  private var tensorBoardLogDir: String = null
  private var tensorBoardAppName: String = null
  private var checkpointPath: String = null
  private var overWriteCheckPoint: Boolean = true
  private var constantGradientClippingParams: (Float, Float) = null
  private var clipNorm: Option[Float] = None

  /**
   * Configure the learning process. It MUST be called before fit or evaluate.
   *
   * @param optimizer Optimization method to be used.
   * @param loss Criterion to be used.
   * @param metrics Validation method(s) to be used. Default is null if no validation is needed.
   */
  def compile(
      optimizer: OptimMethod[T],
      loss: Criterion[T],
      metrics: List[ValidationMethod[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    LoggerFilter.redirectSparkInfoLogs()
    this.optimMethod = optimizer
    this.criterion = loss
    this.vMethods = if (metrics == null) null else metrics.toArray
  }

  /**
   * Alternatively, one can pass in the corresponding Keras-Style
   * string representations when calling compile.
   *
   * For example: optimizer = "sgd", loss = "mse", metrics = List("accuracy")
   */
  def compile(
      optimizer: String,
      loss: String,
      metrics: List[String])(implicit ev: TensorNumeric[T]): Unit = {
    this.compile(KerasUtils.toBigDLOptimMethod[T](optimizer),
      KerasUtils.toBigDLCriterion[T](loss),
      KerasUtils.toBigDLMetrics[T](metrics))
  }

  def compile(
      optimizer: String,
      loss: String)(implicit ev: TensorNumeric[T]): Unit = {
    this.compile(optimizer, loss, null)
  }

  def compile(
      optimizer: OptimMethod[T],
      loss: (Variable[T], Variable[T]) => Variable[T],
      metrics: List[ValidationMethod[T]])(implicit ev: TensorNumeric[T]): Unit = {
    LoggerFilter.redirectSparkInfoLogs()
    val yReal = Variable(KerasUtils.removeBatch(this.getOutputShape()))
    val yPred = Variable(KerasUtils.removeBatch(this.getOutputShape()))
    val lossVar = loss(yReal, yPred)
    val customLoss = new CustomLossWithVariable(Array(yReal, yPred), lossVar)
    this.compile(optimizer, customLoss, metrics)
  }

  def compile(
      optimizer: OptimMethod[T],
      loss: (Variable[T], Variable[T]) => Variable[T])(implicit ev: TensorNumeric[T]): Unit = {
    this.compile(optimizer, loss, null)
  }

  /**
   * Set summary information during the training process for visualization purposes.
   * Saved summary can be viewed via TensorBoard.
   * In order to take effect, it needs to be called before fit.
   *
   * Training summary will be saved to 'logDir/appName/train'
   * and validation summary (if any) will be saved to 'logDir/appName/validation'.
   *
   * @param logDir The base directory path to store training and validation logs.
   * @param appName The name of the application.
   */
  def setTensorBoard(
      logDir: String,
      appName: String): Unit = {
    this.tensorBoardLogDir = logDir
    this.tensorBoardAppName = appName
  }

  /**
   * Configure checkpoint settings to write snapshots every epoch during the training process.
   * In order to take effect, it needs to be called before fit.
   *
   * @param path The path to save snapshots. Make sure this path exists beforehand.
   * @param overWrite Whether to overwrite existing snapshots in the given path. Default is true.
   */
  def setCheckpoint(path: String, overWrite: Boolean = true): Unit = {
    this.checkpointPath = path
    this.overWriteCheckPoint = overWrite
  }

  /**
   * Clear gradient clipping parameters. In this case, gradient clipping will not be applied.
   * In order to take effect, it needs to be called before fit.
   */
  def clearGradientClipping(): Unit = {
    this.constantGradientClippingParams = null
    this.clipNorm = None
  }

  /**
   * Set constant gradient clipping during the training process.
   * In order to take effect, it needs to be called before fit.
   *
   * @param min The minimum value to clip by. Double.
   * @param max The maximum value to clip by. Double.
   */
  def setConstantGradientClipping(min: Float, max: Float): Unit = {
    this.constantGradientClippingParams = (min, max)
  }

  /**
   * Clip gradient to a maximum L2-Norm during the training process.
   * In order to take effect, it needs to be called before fit.
   *
   * @param clipNorm Gradient L2-Norm threshold. Double.
   */
  def setGradientClippingByL2Norm(clipNorm: Float): Unit = {
    this.clipNorm = Some(clipNorm)
  }

  private def toDataSet(x: RDD[Sample[T]], batchSize: Int): DataSet[MiniBatch[T]] = {
    if (x != null) DataSet.rdd(x) -> SampleToMiniBatch[T](batchSize)
    else null
  }

  /**
   * Train a model for a fixed number of epochs on a dataset.
   *
   * @param x Training dataset. If x is an instance of LocalDataSet, train in local mode.
   * @param nbEpoch Number of iterations to train.
   * @param validationData Dataset for validation, or null if validation is not configured.
   */
  def fit[D: ClassTag](
      x: DataSet[D],
      nbEpoch: Int,
      validationData: DataSet[MiniBatch[T]])(implicit ev: TensorNumeric[T]): Unit = {
    require(this.optimMethod != null && this.criterion != null,
      "compile must be called before fit")
    val optimizer = Optimizer(
      model = this,
      dataset = x,
      criterion = this.criterion)
    if (this.checkpointPath != null) {
      optimizer.setCheckpoint(this.checkpointPath, Trigger.everyEpoch)
      if (this.overWriteCheckPoint) {
        optimizer.overWriteCheckpoint()
      }
    }
    if (this.tensorBoardLogDir != null && this.tensorBoardAppName != null) {
      optimizer.setTrainSummary(TrainSummary(tensorBoardLogDir, tensorBoardAppName))
    }
    if (this.constantGradientClippingParams != null) {
      optimizer.setConstantGradientClipping(this.constantGradientClippingParams._1,
        this.constantGradientClippingParams._2)
    }
    if (this.clipNorm.isDefined) {
      optimizer.setGradientClippingByl2Norm(this.clipNorm.get)
    }
    if (validationData != null) {
      require(this.vMethods != null, "Validation metrics haven't been set yet")
      if (this.tensorBoardLogDir != null && this.tensorBoardAppName != null) {
        optimizer.setValidationSummary(ValidationSummary(tensorBoardLogDir, tensorBoardAppName))
      }
      optimizer.setValidation(trigger = Trigger.everyEpoch,
        dataset = validationData,
        vMethods = this.vMethods)
    }
    optimizer.setOptimMethod(this.optimMethod.clone())
      .setEndWhen(Trigger.maxEpoch(nbEpoch))
    optimizer.optimize()
  }

  def fit[D: ClassTag](
      x: DataSet[D],
      nbEpoch: Int)(implicit ev: TensorNumeric[T]): Unit = {
    this.fit(x, nbEpoch, null)
  }

  /**
   * Train a model for a fixed number of epochs on a dataset.
   *
   * @param x Training dataset, RDD of Sample.
   * @param batchSize Number of samples per gradient update. Default is 32.
   * @param nbEpoch Number of iterations to train. Default is 10.
   * @param validationData RDD of Sample, or null if validation is not configured. Default is null.
   */
  def fit(
      x: RDD[Sample[T]],
      batchSize: Int = 32,
      nbEpoch: Int = 10,
      validationData: RDD[Sample[T]] = null)(implicit ev: TensorNumeric[T]): Unit = {
    this.fit(toDataSet(x, batchSize), nbEpoch, toDataSet(validationData, batchSize))
  }

  /**
   * Evaluate a model on a given dataset.
   *
   * @param x Evaluation dataset, RDD of Sample.
   * @param batchSize Number of samples per batch.
   */
  def evaluate(
      x: RDD[Sample[T]],
      batchSize: Int)
      (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    require(this.vMethods != null, "Evaluation metrics haven't been set yet")
    this.evaluate(x, this.vMethods, Some(batchSize))
  }

  /**
   * Evaluate a model in local mode.
   *
   * @param x Evaluation dataset, LocalDataSet.
   */
  def evaluate(x: LocalDataSet[MiniBatch[T]])
      (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    require(this.vMethods != null, "Evaluation metrics haven't been set yet")
    this.evaluate(x, this.vMethods)
  }

  /**
   * Use a model to do prediction.
   *
   * @param x Prediction data, RDD of Sample.
   * @param batchSize Number of samples per batch.
   */
  def predict(
      x: RDD[Sample[T]],
      batchSize: Int)(implicit ev: TensorNumeric[T]): RDD[Activity] = {
    this.predict(x, batchSize, false)
  }

  /**
   * Use a model to do prediction in local mode.
   *
   * @param x Prediction data, LocalDataSet.
   */
  def predict(x: LocalDataSet[MiniBatch[T]])(implicit ev: TensorNumeric[T]): Array[Activity] = {
    val localPredictor = LocalPredictor(this)
    localPredictor.predict(x)
  }

  def toModel(): Model[T]

  /**
   * Print out the summary information of an Analytics Zoo Keras Model.
   *
   * For each layer in the model, there will be a separate row containing four columns:
   * ________________________________________________________________________________
   * Layer (type)          Output Shape          Param #     Connected to
   * ================================================================================
   *
   * In addition, total number of parameters of this model, separated into trainable and
   * non-trainable counts, will be printed out after the table.
   *
   * @param lineLength The total length of one row. Default is 120.
   * @param positions The maximum absolute length proportion(%) of each field.
   *                  Array of Double of length 4.
   *                  Usually you don't need to adjust this parameter.
   *                  Default is Array(.33, .55, .67, 1), meaning that
   *                  the first field will occupy up to 33% of lineLength,
   *                  the second field will occupy up to (55-33)% of lineLength,
   *                  the third field will occupy up to (67-55)% of lineLength,
   *                  the fourth field will occupy the remaining line (100-67)%.
   *                  If the field has a larger length, the remaining part will be trimmed.
   *                  If the field has a smaller length, the remaining part will be white spaces.
   */
  def summary(
      lineLength: Int = 120,
      positions: Array[Double] = Array(.33, .55, .67, 1)): Unit
}

class Model[T: ClassTag] private (private val _inputs : Seq[ModuleNode[T]],
    private val _outputs : Seq[ModuleNode[T]])(implicit ev: TensorNumeric[T])
  extends KerasNet[T] with NetUtils[T, Model[T]] {
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
    "com.intel.analytics.zoo.pipeline.api.keras.models.Model",
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

class Sequential[T: ClassTag] private ()
  (implicit ev: TensorNumeric[T]) extends KerasNet[T] {

  private[zoo] var frozen: Boolean = false

  this.labor = doBuild(null)

  private def buildModule(module: AbstractModule[_ <: Activity, _ <: Activity, T]): Unit = {
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

  private def getLambdaLayer(lambda: Lambda[T]):
  AbstractModule[_ <: Activity, _ <: Activity, T] = {
    val inputShape = if (!this.isBuilt()) {
      if (lambda.getInputShape() == null) {
        throw new RuntimeException("The first layer should explicitly declare inputshape")
      }
      lambda.getInputShape()
    } else {
      this.getOutputShape()
    }
    return lambda.create(
      KerasUtils.removeBatch(inputShape))
  }

  def add(lambda: Lambda[T]): Sequential[T] = {
    add(getLambdaLayer(lambda))
  }

  /**
   * Add a sub-module to the sequential container.
   *
   * @param module The module to be added.
   * @return This sequential container.
   */
  def add(module: AbstractModule[_ <: Activity, _ <: Activity, T]): this.type = {
    if (frozen) {
      throw new RuntimeException(
        "This Sequential has been frozen, as it has been added into other container")
    }

    if (module.isInstanceOf[Sequential[T]]) {
      module.asInstanceOf[Sequential[T]].frozen = true
    }
    val mModule = module
    val kerasLayerRef = KerasLayerRef(this)
    kerasLayerRef.validateInput[T](Seq(mModule))

    buildModule(mModule)

    labor.asInstanceOf[TSequential[T]].modules +=
      mModule.asInstanceOf[AbstractModule[Activity, Activity, T]]
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

  override def toModel(): Model[T] = {
    val input = Input[T](KerasUtils.removeBatch(this.getInputShape()))

    // the is reason we do not use .inputs here is
    // layers in modules cannot be rebuilt
    val output = this.modules(0)
      .asInstanceOf[TSequential[T]]
      .modules.foldLeft(input) { (i1, i2) =>
      val out = Node(i2)
      i1.add(out, Edge())
      out
    }
    Model(input, output)
  }

  override def summary(
      lineLength: Int = 120,
      positions: Array[Double] = Array(.33, .55, .67, 1)): Unit = {
    val graph = this.toModel()
    graph.summary(lineLength, positions)
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
