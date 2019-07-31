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

import java.io.{File, FilenameFilter}
import java.text.SimpleDateFormat
import java.util.Calendar

import com.intel.analytics.bigdl.dataset.{MiniBatch, _}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.{DataSet, _}
import com.intel.analytics.bigdl.nn.Graph._
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.keras.{KerasLayer, KerasLayerSerializable}
import com.intel.analytics.bigdl.nn.mkldnn.MklDnnModule
import com.intel.analytics.bigdl.nn.{Container, Graph, Module, StaticGraph, Sequential => TSequential}
import com.intel.analytics.bigdl.optim.DistriOptimizer.Cache
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.parameters.AllReduceParameter
import com.intel.analytics.bigdl.serialization.Bigdl.BigDLModule
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.utils._
import com.intel.analytics.bigdl.utils.serializer.{DeserializeContext, ModuleData, ModuleSerializer, SerializeContext}
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.common.ZooTrigger
import com.intel.analytics.zoo.feature.{DiskFeatureSet, DistributedFeatureSet, FeatureSet}
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.feature.text._
import com.intel.analytics.zoo.pipeline.api.{Net, Predictable}
import com.intel.analytics.zoo.pipeline.api.autograd.{Lambda, Variable}
import com.intel.analytics.zoo.pipeline.api.autograd._
import com.intel.analytics.zoo.pipeline.api.keras.layers.Input
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils._
import com.intel.analytics.zoo.pipeline.api.net.NetUtils
import com.intel.analytics.zoo.pipeline.estimator.{AbstractEstimator, ConstantClipping, GradientClipping, L2NormClipping}
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.log4j.Logger
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.language.implicitConversions

abstract class KerasNet[T](implicit val tag: ClassTag[T], implicit val ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T] with Net with Predictable[T] {

  protected val module: Module[T] = this

  def getSubModules(): List[AbstractModule[Activity, Activity, T]] = {
    require(this.labor.isInstanceOf[Container[Activity, Activity, T]],
      s"labor should be a container, but we got: $this")
    this.labor.asInstanceOf[Container[Activity, Activity, T]].modules.toList
  }

  private var optimMethod: OptimMethod[T] = null
  @transient private var internalOptimizer: Optimizer[T, MiniBatch[T]] = null
  private var criterion: Criterion[T] = null
  private var vMethods: Array[ValidationMethod[T]] = null
  private var tensorBoardLogDir: String = null
  private var tensorBoardAppName: String = null
  @transient private var trainSummary: TrainSummary = null
  @transient private var validationSummary: ValidationSummary = null
  private var checkpointPath: String = null
  private var overWriteCheckPoint: Boolean = true
  private var constantGradientClippingParams: (Float, Float) = null
  private var clipNorm: Option[Float] = None

  private def getOrCreateOptimizer(x: DataSet[MiniBatch[T]]): Optimizer[T, MiniBatch[T]] = {
    if (null != this.internalOptimizer) {
      return internalOptimizer
    }
    this.internalOptimizer = x match {
      case local: LocalDataSet[MiniBatch[T]] =>
        new InternalLocalOptimizer(model = this,
          ds = local,
          criterion = this.criterion)
      case distriDataSet: DistributedDataSet[MiniBatch[T]] =>
        new InternalDistriOptimizer(_model = this,
          _dataset = distriDataSet,
          _criterion = this.criterion)
      case distriFeatureSet: DistributedFeatureSet[MiniBatch[T]] =>
        new InternalDistriOptimizer(_model = this,
          _dataset = distriFeatureSet.toDistributed(),
          _criterion = this.criterion)
      case _ =>
        throw new IllegalArgumentException(s"Unsupported DataSet type ${x.getClass.getName}," +
          s" excepted LocalDataSet, DistributedDataSet and DistributedFeatureSet.")
    }

    if (this.checkpointPath != null) {
      internalOptimizer.setCheckpoint(this.checkpointPath, Trigger.everyEpoch)
      if (this.overWriteCheckPoint) {
        internalOptimizer.overWriteCheckpoint()
      }
    }
    if (this.tensorBoardLogDir != null && this.tensorBoardAppName != null) {
      this.trainSummary = TrainSummary(tensorBoardLogDir, tensorBoardAppName)
      internalOptimizer.setTrainSummary(this.trainSummary)
    }
    if (this.constantGradientClippingParams != null) {
      internalOptimizer.setConstantGradientClipping(this.constantGradientClippingParams._1,
        this.constantGradientClippingParams._2)
    }
    if (this.clipNorm.isDefined) {
      internalOptimizer.setGradientClippingByl2Norm(this.clipNorm.get)
    }
    this.internalOptimizer
  }
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

    val lossArray: Array[ValidationMethod[T]] = Array(new Loss(this.criterion))

    if (metrics == null) {
      this.vMethods = lossArray
    }
    else {
      val metricsArray = metrics.toArray
      this.vMethods = lossArray ++ metricsArray
    }
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
      KerasUtils.toBigDLMetrics[T](metrics, loss))
  }

  def compile(
      optimizer: String,
      loss: String)(implicit ev: TensorNumeric[T]): Unit = {
    this.compile(optimizer, loss, null)
  }

  /**
   * You can also use custom loss function during compile.
   */
  def compile(
      optimizer: OptimMethod[T],
      loss: (Variable[T], Variable[T]) => Variable[T],
      metrics: List[ValidationMethod[T]])(implicit ev: TensorNumeric[T]): Unit = {
    LoggerFilter.redirectSparkInfoLogs()
    val customLoss = CustomLoss[T](loss, KerasUtils.removeBatch(this.getOutputShape()))
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
  def setTensorBoard(logDir: String, appName: String): Unit = {
    if (this.internalOptimizer != null) {
      this.trainSummary = TrainSummary(tensorBoardLogDir, tensorBoardAppName)
      internalOptimizer.setTrainSummary(this.trainSummary)
    }
    this.tensorBoardLogDir = logDir
    this.tensorBoardAppName = appName
  }

  /**
   * To get the scalar like "Loss", "LearningRate" from train summary
   * Return is a Array of 3-tuples
   *
   * @param tag The string variable represents the parameter you want to return
   *            supported tags are "LearningRate", "Loss", "Throughput"
   */
  def getTrainSummary(tag: String): Array[(Long, Float, Double)] = {
    this.trainSummary.readScalar(tag)
  }

  /**
   * To get the scalar like "Loss", "Top1Accuracy" from validation summary
   * Return is a Array of 3-tuples
   *
   * @param tag The string variable represents the parameter you want to return
   *            supported tags are 'AUC', 'Accuracy', 'BinaryAccuracy', 'CategoricalAccuracy',
    *           'HitRatio', 'Loss', 'MAE', 'NDCG', 'SparseCategoricalAccuracy',
    *           'TFValidationMethod', 'Top1Accuracy',
    *           'Top5Accuracy', 'TreeNNAccuracy'.
   */
  def getValidationSummary(tag: String): Array[(Long, Float, Double)] = {
    this.validationSummary.readScalar(tag)
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

    if (this.internalOptimizer != null) {
      internalOptimizer.setCheckpoint(this.checkpointPath, Trigger.everyEpoch)
      if (this.overWriteCheckPoint) {
        internalOptimizer.overWriteCheckpoint()
      }
    }
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
    if (this.internalOptimizer != null) {
      internalOptimizer.setConstantGradientClipping(min, max)
    }
    this.constantGradientClippingParams = (min, max)
  }

  /**
   * Clip gradient to a maximum L2-Norm during the training process.
   * In order to take effect, it needs to be called before fit.
   *
   * @param clipNorm Gradient L2-Norm threshold. Double.
   */
  def setGradientClippingByL2Norm(clipNorm: Float): Unit = {
    if (this.internalOptimizer != null) {
      this.internalOptimizer.setGradientClippingByl2Norm(clipNorm)
    }
    this.clipNorm = Some(clipNorm)
  }

  /**
   * Set the model to be in evaluate status, i.e. remove the effect of Dropout, etc.
   */
  def setEvaluateStatus(): this.type = {
    evaluate()
  }

  /**
   * Convert RDD of Sample to DataSet of MiniBatch.
   */
  private def toDataSet(x: RDD[Sample[T]], batchSize: Int,
    featurePaddingParam: PaddingParam[T] = null,
    labelPaddingParam: PaddingParam[T] = null): DataSet[MiniBatch[T]] = {
    val _featurePaddingParam = if (featurePaddingParam != null) {
      Some(featurePaddingParam)
    } else None
    val _labelPaddingParam = if (labelPaddingParam != null) {
      Some(labelPaddingParam)
    } else None

    if (x != null) DataSet.rdd(x) -> SampleToMiniBatch[T](batchSize, _featurePaddingParam,
      _labelPaddingParam)
    else null
  }

  /**
   * Convert ImageSet to DataSet of MiniBatch.
   */
  private def toDataSet(x: ImageSet, batchSize: Int): DataSet[MiniBatch[T]] = {
    if (x != null) x.toDataSet[T]() -> SampleToMiniBatch[T](batchSize)
    else null
  }

  /**
   * Convert TextSet to DataSet of MiniBatch.
   */
  private def toDataSet(x: TextSet, batchSize: Int): DataSet[MiniBatch[T]] = {
    if (x != null) {
      (x.toDataSet -> SampleToMiniBatch[Float](batchSize)).asInstanceOf[DataSet[MiniBatch[T]]]
    }
    else null
  }

  /**
   * Train a model for a fixed number of epochs on a DataSet.
   *
   * @param x Training dataset. If x is an instance of LocalDataSet, train in local mode.
   * @param nbEpoch Number of epochs to train.
   * @param validationData Dataset for validation, or null if validation is not configured.
   */
  def fit(
      x: DataSet[MiniBatch[T]],
      nbEpoch: Int,
      validationData: DataSet[MiniBatch[T]])(implicit ev: TensorNumeric[T]): Unit = {
    require(this.optimMethod != null && this.criterion != null,
      "compile must be called before fit")
    this.internalOptimizer = this.getOrCreateOptimizer(x)
    if (validationData != null) {
      require(this.vMethods != null, "Validation metrics haven't been set yet")
      if (this.tensorBoardLogDir != null && this.tensorBoardAppName != null) {
        this.validationSummary = ValidationSummary(tensorBoardLogDir, tensorBoardAppName)
        internalOptimizer.setValidationSummary(this.validationSummary)
      }
      internalOptimizer.setValidation(trigger = Trigger.everyEpoch,
        dataset = validationData,
        vMethods = this.vMethods)
    }
    internalOptimizer.setOptimMethod(this.optimMethod)
      .setEndWhen(Trigger.maxEpoch(getFinishedEpoch() + nbEpoch))

    internalOptimizer match {
      case local: InternalLocalOptimizer[T] =>
        local.setTrainData(x)
      case dis: InternalDistriOptimizer[T] =>
        dis.setTrainData(x)
    }

    internalOptimizer.optimize()
  }

  private def getFinishedEpoch() = {
    internalOptimizer match {
      // epoch# from optimizer and optimMethod is not consistent in BigDL.
      case local: LocalOptimizer[T] =>
        val state = InternalOptimizerUtil.getStateFromOptimizer(this.internalOptimizer)
        if (state.get[Int]("epoch").isDefined) {
          state.get[Int]("epoch").get - 1
        } else {
          0
        }
      case dis: DistriOptimizer[T] =>
        InternalOptimizerUtil.getStateFromOptiMethod(this.optimMethod).get[Int]("epoch").get - 1
    }
  }

  def fit(
      x: DataSet[MiniBatch[T]],
      nbEpoch: Int)(implicit ev: TensorNumeric[T]): Unit = {
    this.fit(x, nbEpoch, null)
  }

  /**
   * Release DataSet from memory. This method is used to release the rdd
   * which is cached when toDataSet() method is called and rdd is cached
   * TODO: modify this when BigDL fix this issue
   *
   * @param dataSet Target DataSet to release
   */
  def releaseDataSets(dataSets: Array[DataSet[MiniBatch[T]]]): Unit = {
    for (ds <- dataSets) {
      if (ds != null && ds.isInstanceOf[DistributedDataSet[T]]) {
        ds.toDistributed().unpersist()
        ds.toDistributed().originRDD().unpersist()
      }
    }
  }

  /**
   * Train a model for a fixed number of epochs on Sample RDD.
   *
   * @param x Training dataset, RDD of Sample.
   * @param batchSize Number of samples per gradient update. Default is 32.
   * @param nbEpoch Number of epochs to train. Default is 10.
   * @param validationData RDD of Sample, or null if validation is not configured. Default is null.
   */
  def fit(
      x: RDD[Sample[T]],
      batchSize: Int = 32,
      nbEpoch: Int = 10,
      validationData: RDD[Sample[T]] = null,
      featurePaddingParam: PaddingParam[T] = null,
      labelPaddingParam: PaddingParam[T] = null)(implicit ev: TensorNumeric[T]): Unit = {
    KerasUtils.validateBatchSize(batchSize)
    val trainData = toDataSet(x, batchSize, featurePaddingParam, labelPaddingParam)
    val valData = toDataSet(validationData, batchSize, featurePaddingParam, labelPaddingParam)
    this.fit(trainData, nbEpoch, valData)

    releaseDataSets(Array(trainData, valData))
  }

  /**
   * Train a model for a fixed number of epochs on ImageSet.
   *
   * @param x Training dataset, ImageSet.
   * @param batchSize Number of samples per gradient update.
   * @param nbEpoch Number of epochs to train.
   * @param validationData ImageSet, or null if validation is not configured.
   */
  def fit(
      x: ImageSet,
      batchSize: Int,
      nbEpoch: Int,
      validationData: ImageSet)(implicit ev: TensorNumeric[T]): Unit = {
    KerasUtils.validateBatchSize(batchSize)
    val trainData = toDataSet(x, batchSize)
    val valData = toDataSet(validationData, batchSize)

    this.fit(trainData, nbEpoch, valData)
    releaseDataSets(Array(trainData, valData))
  }

  def fit(
      x: ImageSet,
      batchSize: Int,
      nbEpoch: Int)(implicit ev: TensorNumeric[T]): Unit = {
    this.fit(x, batchSize, nbEpoch, null)
  }

  /**
   * Train a model for a fixed number of epochs on TextSet.
   *
   * @param x Training dataset, TextSet.
   * @param batchSize Number of samples per gradient update.
   * @param nbEpoch Number of epochs to train.
   * @param validationData TextSet, or null if validation is not configured.
   */
  def fit(
      x: TextSet,
      batchSize: Int,
      nbEpoch: Int,
      validationData: TextSet)(implicit ev: TensorNumeric[T]): Unit = {
    KerasUtils.validateBatchSize(batchSize)
    val dataset = x.toDataSet
    this.fit((dataset -> SampleToMiniBatch[Float](batchSize)).asInstanceOf[DataSet[MiniBatch[T]]],
      nbEpoch, toDataSet(validationData, batchSize))
    if (x.isDistributed) {
      dataset.toDistributed().unpersist()
    }
  }

  def fit(
      x: TextSet,
      batchSize: Int,
      nbEpoch: Int)(implicit ev: TensorNumeric[T]): Unit = {
    this.fit(x, batchSize, nbEpoch, null)
  }

  /**
   * Evaluate a model on given RDD.
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
   * Evaluate a model on ImageSet.
   *
   * @param x Evaluation dataset, ImageSet.
   * @param batchSize Number of samples per batch.
   */
  def evaluate(
      x: ImageSet,
      batchSize: Int)
      (implicit ev: TensorNumeric[T]): Array[(ValidationResult, ValidationMethod[T])] = {
    require(this.vMethods != null, "Evaluation metrics haven't been set yet")
    evaluateImage(x.toImageFrame(), this.vMethods, Some(batchSize))
  }

  /**
   * Evaluate a model on TextSet.
   *
   * @param x Evaluation dataset, TextSet.
   * @param batchSize Number of samples per batch.
   */
  def evaluate(
      x: TextSet,
      batchSize: Int): Array[(ValidationResult, ValidationMethod[T])] = {
    require(this.vMethods != null, "Evaluation metrics haven't been set yet")
    x match {
      case distributed: DistributedTextSet =>
        val rdd = distributed.rdd.map(_.getSample).filter(_ != null)
        evaluate(rdd.asInstanceOf[RDD[Sample[T]]], batchSize)
      case local: LocalTextSet =>
        val localSet = toDataSet(local, batchSize).asInstanceOf[LocalDataSet[MiniBatch[T]]]
        evaluate(localSet)
    }
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

object Sequential extends KerasLayerSerializable {
  ModuleSerializer.registerModule(
    "com.intel.analytics.zoo.pipeline.api.keras.models.Sequential",
    Sequential)

  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : Sequential[T] = {
    new Sequential[T]()
  }
}

private[zoo] object InternalOptimizerUtil {


  def getModelCacheFromOptimizer[T: ClassTag](
        optimizer: Optimizer[T, MiniBatch[T]]): RDD[Cache[T]] = {
    val field = classOf[DistriOptimizer[T]].getDeclaredField("models")
    field.setAccessible(true)
    val models = field.get(optimizer).asInstanceOf[RDD[Cache[T]]]
    models
  }

  def getStateFromOptiMethod[T](optimMethod: OptimMethod[T]): Table = {
    val method = classOf[OptimMethod[T]].getDeclaredMethod("state")
    method.setAccessible(true)
    val state = method.invoke(optimMethod).asInstanceOf[Table]
    state
  }

  def getStateFromOptimizer[T: ClassTag](optimizer: Optimizer[T, MiniBatch[T]]): Table = {
    val method = classOf[Optimizer[T, MiniBatch[T]]].getDeclaredMethod("state")
    method.setAccessible(true)
    val state = method.invoke(optimizer).asInstanceOf[Table]
    state
  }

  def endEpoch[T: ClassTag](optimizer: DistriOptimizer[T]): Unit = {
    val method = classOf[DistriOptimizer[T]].getDeclaredMethod("endEpoch")
    method.setAccessible(true)
    method.invoke(optimizer)
  }

  def getParametersFromModel[T: ClassTag](model: Module[T]): (Tensor[T], Tensor[T]) = {
    val method = classOf[Module[T]].getDeclaredMethod("getParameters")
    method.setAccessible(true)
    method.invoke(model).asInstanceOf[(Tensor[T], Tensor[T])]
  }

  def initThreadModels[T: ClassTag](
      args: Object*)(
      implicit ev: TensorNumeric[T]): (RDD[DistriOptimizer.Cache[T]], ModelBroadcast[T]) = {
    KerasUtils.invokeMethodWithEv(DistriOptimizer,
      "com$intel$analytics$bigdl$optim$DistriOptimizer$$initThreadModels",
      args: _*).asInstanceOf[(RDD[DistriOptimizer.Cache[T]], ModelBroadcast[T])]
  }

  def clearState[T: ClassTag](
        models: RDD[DistriOptimizer.Cache[T]]): Unit = {
    KerasUtils.invokeMethod(DistriOptimizer,
      "clearState", models, implicitly[reflect.ClassTag[T]])
  }

  def optimizeModels[T: ClassTag](
      args: Object*
      )(implicit ev: TensorNumeric[T]): Unit = {
    KerasUtils.invokeMethodWithEv(DistriOptimizer, "optimize",
      args: _*)
  }

  def getModel[T: ClassTag](
      args: Object*)(implicit ev: TensorNumeric[T]): Unit = {
    KerasUtils.invokeMethodWithEv(DistriOptimizer, "getModel",
      args: _*)
  }

  def releaseBroadcast[T: ClassTag](
        uuid: String)(implicit ev: TensorNumeric[T]): Unit = {
    KerasUtils.invokeMethodWithEv(
      "com.intel.analytics.bigdl.models.utils.CachedModels",
      "deleteKey",
      uuid)
  }

}

private[zoo] class InternalLocalOptimizer[T: ClassTag] (
    model: Module[T],
    ds: LocalDataSet[MiniBatch[T]],
    criterion: Criterion[T])
  (implicit ev: TensorNumeric[T]) extends LocalOptimizer[T](model, ds, criterion) {

  def setTrainData(trainingDataSet: DataSet[MiniBatch[T]]): this.type = {
    this.dataset = trainingDataSet
    this.endEpoch()
    this
  }

  // LocalOptimizer use this `optimizer.state` to control the training
  // But there's no logic to update the "recordsProcessedThisEpoch"
  // neither in optimizer.state nor optimMethod.state.
  // So we can only simply suppose the `epoch` has been correctly updated.
  def endEpoch[T: ClassTag](): Unit = {
  }
}

private[zoo] class InternalDistriOptimizer[T: ClassTag] (
    _model: Module[T],
    _dataset: DistributedDataSet[MiniBatch[T]],
    _criterion: Criterion[T])
  (implicit ev: TensorNumeric[T]) extends DistriOptimizer[T](_model, _dataset, _criterion)
  with AbstractEstimator[T]{
  import InternalDistriOptimizer._
  protected var checkpointDir: Option[String] = None
  protected var numSlice: Int = 1
  protected var cachedModels: RDD[DistriOptimizer.Cache[T]] = null
  protected var modelBroadcast: ModelBroadcast[T] = null
  protected var parameterSplits: Map[String, (Int, Int)] = null
  protected var allReduceParameter: AllReduceParameter[T] = null

  def train(): Module[T] = {
    val distDataset = dataset.toDistributed()
    val trainingModel = if (EngineRef.getEngineType() == MklDnn) {
      model.toGraph().setName(model.getName())
    } else model

    // To be compatible with the old usage that user define hyperparameters in a table.
    if (optimMethods.size == 1) {
      optimMethods.head._2.loadFromTable(state)
    }

    state("dropPercentage") = dropPercentage
    state("warmupIterationNum") = warmupIterationNum
    state("computeThresholdbatchSize") = computeThresholdbatchSize
    state("maxDropPercentage") = maxDropPercentage
    state("isLayerwiseScaled") = com.intel.analytics.bigdl.nn.Utils.isLayerwiseScaled(_model)

    val nodeNumber = EngineRef.getNodeNumber()
    val coresPerNode = EngineRef.getCoreNumber()

    val partitionNum = distDataset.originRDD().partitions.length
    val modelParameters = InternalOptimizerUtil.getParametersFromModel(trainingModel)

    prepareInput()

    // subModuleName -> (storageOffset, length, AllReduceParameter)
    if (allReduceParameter == null || cachedModels == null) {
      allReduceParameter = AllReduceParameter.newParameter[T](partitionNum,
        modelParameters._1.nElement())
      this.close()
      parameterSplits = if (optimMethods.size != 1) {
        val p = optimMethods.map { case (subModuleName, optimMethod) =>
          val subModule = trainingModel(subModuleName)
          require(subModule.isDefined, s"Optimizer couldn't find $subModuleName in $model")
          val subModuleWeights = InternalOptimizerUtil
            .getParametersFromModel(subModule.get)._1
          (subModuleName, subModuleWeights)
        }
        val sortedWeights = p.values.toArray.sortWith(
          (a, b) => a.storageOffset() < b.storageOffset())
        val compactWeights = Module.isCompact(sortedWeights)
        require(modelParameters._1 == compactWeights,
          s"InternDistriOptimizer: All subModules should have an OptimMethod.")
        p.map { case (subModuleName, weights) =>
          (subModuleName, (weights.storageOffset(), weights.nElement()))
        }
      } else if (optimMethods.contains(trainingModel.getName())) {
        Map(trainingModel.getName() -> (1, modelParameters._1.nElement()))
      } else {
        throw new IllegalArgumentException(s"${trainingModel.getName()} doesn't " +
          s"have corresponding OptimMethod")
      }

      // TODO: Enable LarsSGD
//      LarsSGD.containsLarsSGD(optimMethods).foreach(weightDecay =>
//        parameterProcessors.append(new LarsProcessor(parameterSplits, weightDecay))
//      )

      val modelsAndBroadcast = InternalOptimizerUtil.initThreadModels[T](
        trainingModel, distDataset, criterion, state,
        Int.box(nodeNumber), Int.box(coresPerNode), Boolean.box(checkSingleton),
        allReduceParameter, parameterSplits, validationMethods, optimMethods, parameterProcessors)
      cachedModels = modelsAndBroadcast._1
      modelBroadcast = modelsAndBroadcast._2
    }

    val currentCheckPoint = if (checkpointPath.isDefined) {
      val file = checkpointPath.get + "/" +
        new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime())
      new File(file).mkdir()
      Some(file)
    } else {
      checkpointPath
    }


    var retryNum = 0
    val maxRetry = System.getProperty("bigdl.failure.retryTimes", "5").toInt
    val retryTimeInterval = System.getProperty("bigdl.failure.retryTimeInterval", "120").toInt
    var lastFailureTimestamp = System.nanoTime()

    while (retryNum < maxRetry) {
      try {
        InternalOptimizerUtil.optimizeModels[T](
          trainingModel,
          distDataset,
          Int.box(coresPerNode),
          state,
          endWhen,
          metrics,
          cachedModels,
          optimMethods,
          allReduceParameter,
          parameterSplits,
          validationTrigger,
          validationDataSet,
          validationMethods,
          checkpointTrigger,
          currentCheckPoint,
          trainSummary,
          validationSummary,
          Boolean.box(isOverWrite),
          parameterProcessors.toArray)
        retryNum = Int.MaxValue
      } catch {
        case e: IllegalArgumentException =>
          throw e
        case t: Throwable =>
          DistriOptimizer.logger.error("Error: " + ExceptionUtils.getStackTrace(t))
          if (checkpointPath.isDefined) {
            /* To avoid retry number is used up by first few exceptions, we count time here.
             * If exception exceeds maxRetry times in maxRetry*retryTimeInterval seconds,
             * we will give up retry Or we will reset retryNum
             */
            if (System.nanoTime() - lastFailureTimestamp < maxRetry * retryTimeInterval * 1e9) {
              retryNum += 1
              if (retryNum == maxRetry) {
                throw t
              }
            } else {
              retryNum = 1
            }
            DistriOptimizer.logger.info(s"Retrying $retryNum times")
            lastFailureTimestamp = System.nanoTime()

            val modelFile = getLatestFile(currentCheckPoint.get, "model")
            clearState()
            cachedModels.unpersist()
            val newModel = if (modelFile != null) {
              DistriOptimizer.logger.info("Model recover from last snapshot")
              Module.load[T](modelFile)
            } else {
              DistriOptimizer.logger.info("Model recover from origin model")
              trainingModel
            }
            optimMethods = optimMethods.map { case (moduleName, optimMethod) =>
              val methodFile = getLatestFile(currentCheckPoint.get, s"optimMethod-$moduleName")

              val newOptimMethod = if (methodFile != null) {
                DistriOptimizer.logger.info(s"$moduleName's OptimMethod recover from last snapshot")
                OptimMethod.load[T](methodFile)
              } else {
                DistriOptimizer.logger.info(s"$moduleName's OptimMethod recover from origin model")
                optimMethod
              }
              newOptimMethod.clearHistory()
              (moduleName, newOptimMethod)
            }
            val modelsAndBroadcast = InternalOptimizerUtil.initThreadModels[T](
              newModel, distDataset, criterion, state,
              Int.box(nodeNumber), Int.box(coresPerNode), Boolean.box(checkSingleton),
              allReduceParameter, parameterSplits, validationMethods, optimMethods)
            cachedModels = modelsAndBroadcast._1
            modelBroadcast = modelsAndBroadcast._2
          } else {
            throw t
          }
      }
    }

    InternalOptimizerUtil.getModel(
      cachedModels, allReduceParameter, trainingModel)

    trainingModel
  }

  override def close(): Unit = {
    if (cachedModels != null) {
      InternalOptimizerUtil.clearState(cachedModels)
      if (modelBroadcast != null) {
        InternalOptimizerUtil.releaseBroadcast(modelBroadcast.uuid())
        modelBroadcast = null
      }
      unpersistCachedModel(cachedModels)
      cachedModels = null
    }
  }


  def setNumOfSlice(numOfSlice: Int): this.type = {
    require(numOfSlice >= 0, s"excepted numOfSlice >= 0," +
      s" but got $numOfSlice")
    this.numSlice = numOfSlice
    this
  }

  def getNumOfSlice(): Int = {
    this.numSlice
  }

  def setCheckpointDir(path: Option[String]): this.type = {
    if (path.isDefined) {
      val pathAndTime = path.get + "/" +
        new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss")
          .format(Calendar.getInstance().getTime())
      checkpointDir = Some(pathAndTime)
      logger.info(s"Saving summaries to ${pathAndTime + "/summary"}")
      val trainSummary = TrainSummary(pathAndTime, "summary")
      val valSummary = ValidationSummary(pathAndTime, "summary")
      this.setTrainSummary(trainSummary)
      this.setValidationSummary(valSummary)
    }
    this
  }

  def setTrainData(trainingDataSet: DataSet[MiniBatch[T]]): this.type = {
    this.dataset = trainingDataSet.toDistributed()
    InternalOptimizerUtil.endEpoch(this)
    this
  }


  override def train(
        trainSet: FeatureSet[MiniBatch[T]],
        criterion: Criterion[T],
        endTrigger: Option[Trigger] = None,
        checkPointTrigger: Option[Trigger] = None,
        validationSet: FeatureSet[MiniBatch[T]] = null,
        validationMethod: Array[ValidationMethod[T]] = null): this.type = {
    this.dataset = trainSet.toDataSet()
    val endWhen = if (endTrigger.isDefined) {
      endTrigger.get
    } else {
      Trigger.maxIteration(Int.MaxValue)
    }
    this.setEndWhen(endWhen)
    if (checkPointTrigger.isDefined && checkpointDir.isDefined) {
      // we should setCheckpoint every time before we call optimize(),
      // as BigDL will overwrite checkpointPath to its subfolder.
      val logPath = new Path(checkpointDir.get + "/models")
      val fs = logPath.getFileSystem(new Configuration(false))
      fs.mkdirs(logPath)
      logger.info(s"Saving checkpoints to ${logPath.toUri.toString}")
      this.setCheckpoint(logPath.toUri.toString(), checkPointTrigger.get)
    }
    if (checkPointTrigger.isDefined && validationMethod != null && validationSet != null) {
      this.setValidation(checkPointTrigger.get, validationSet.toDataSet(), validationMethod)
    }
    if (numSlice != 1) {
      val state = InternalOptimizerUtil.getStateFromOptiMethod(
        optimMethods.values.head)
      if (checkPointTrigger.isDefined) {
        if (checkPointTrigger.get.isInstanceOf[ZooTrigger]) {
          checkPointTrigger.get.asInstanceOf[ZooTrigger].setZooState(state)
        } else {
          throw new IllegalArgumentException(
            s"Excepted com.intel.analytics.zoo.common.ZooTrigger." +
            s" Please change your trigger to an instance of ZooTrigger.")
        }
      }
      if (!state.contains("numSlice")) {
        state("numSlice") = numSlice
        state("currentSlice") = 0
      }
      if (!state.contains("Loss")) {
        state.update("Loss", Float.PositiveInfinity)
      }
      if (!state.contains("score")) {
        state.update("score", 0f)
      }

      while(!endWhen(state)) {
        val trueEpoch = Math.floor(state[Int]("currentSlice").toDouble / numSlice).toInt + 1
        val newEndWhen = Trigger.or(endWhen, Trigger.maxEpoch(trueEpoch))
        this.setEndWhen(newEndWhen)
        if (checkPointTrigger.isDefined && checkpointDir.isDefined) {
          // we should setCheckpoint every time before we call optimize(),
          // as BigDL will overwrite checkpointPath to its subfolder.
          this.setCheckpoint(checkpointDir.get, checkPointTrigger.get)
        }
        state("currentSlice") = state[Int]("currentSlice") + 1
        this.train()
        InternalOptimizerUtil.endEpoch(this)
        // (neval - 1) is true iteration
        state("epoch") = Math.floor(state[Int]("currentSlice").toDouble / numSlice).toInt + 1
      }
    } else {
      this.train()
    }
    this
  }

  override def evaluate(
        validationSet: FeatureSet[MiniBatch[T]],
        validationMethod: Array[ValidationMethod[T]]
      ): Map[ValidationMethod[T], ValidationResult] = {
    val validateRDD = validationSet.toDistributed().data(train = false)
    val sc = validateRDD.sparkContext
    val cachedModels = InternalOptimizerUtil.getModelCacheFromOptimizer(this)

    val coresPerNode = EngineRef.getCoreNumber()
    val _subModelNumber = EngineRef.getEngineType() match {
      case MklBlas => coresPerNode
      case MklDnn => 1
      case _ => throw new IllegalArgumentException
    }

    val models = if (null != cachedModels) {
      val bcVMethods = cachedModels.sparkContext.broadcast(validationMethod)
      cachedModels.map{cache =>
        Cache[T](
          cache.localModels,
          cache.modelWeights,
          cache.modelGradients,
          cache.localCriterions,
          cache.localStates,
          cache.moduleTimeList,
          Array.tabulate(_subModelNumber)(_ =>
            Some(bcVMethods.value.map(_.clone()))),
          cache.optimMethods,
          cache.parameterSynchronizer
        )
      }
    } else {
      val bcVMethods = validateRDD.sparkContext.broadcast(validationMethod)
      val bcModel = ModelBroadcast[T]().broadcast(sc, model)
      validateRDD.mapPartitions{_ =>
        Iterator.single(Cache[T](
          Array.tabulate(_subModelNumber)(_ => bcModel.value()),
          null,
          null,
          null,
          null,
          null,
          Array.tabulate(_subModelNumber) { _ =>
            Some(bcVMethods.value.map(_.clone()))},
          null,
          null
        ))
      }
    }

    // get current iteration from optimMethod
    val step = if (null != optimMethods && optimMethods.size != 0) {
      val state = InternalOptimizerUtil.getStateFromOptiMethod(
        optimMethods.values.head)
      state.getOrElse[Int]("neval", -1)
    } else {
      -1
    }

    InternalDistriOptimizer.validate(
      validationSet,
      validationMethod,
      models,
      step,
      validationSummary
    )
  }
}

object InternalDistriOptimizer {
  val logger = Logger.getLogger(this.getClass)

  protected def validate[T](validationFeatureSet: FeatureSet[MiniBatch[T]],
                            validationMethods: Array[ValidationMethod[T]],
                            models: RDD[Cache[T]],
                            step: Int,
                            validationSummary: Option[ValidationSummary]
                           ): Map[ValidationMethod[T], ValidationResult] = {
    val vMethods = validationMethods
    val validateRDD = validationFeatureSet.toDistributed().data(train = false)
    val _subModelNumber = EngineRef.getEngineType match {
      case MklBlas => EngineRef.getCoreNumber()
      case MklDnn => 1
      case _ => throw new IllegalArgumentException
    }
    // TODO: evaluate local
    val results = ZippedPartitionsWithLocalityRDD(models, validateRDD)(
      (modelIter, dataIter) => {
        val cached = modelIter.next()
        val workingModels = cached.localModels
        val localVMethods = cached.localMethods

        workingModels.foreach(_.evaluate())
        dataIter.map(batch => {
          val stackSize = batch.size() / _subModelNumber
          val extraSize = batch.size() % _subModelNumber
          val parallelism = if (stackSize == 0) extraSize else _subModelNumber
          (0 until parallelism).toParArray.map { b =>
            val offset = b * stackSize + math.min(b, extraSize) + 1
            val length = stackSize + (if (b < extraSize) 1 else 0)
            val miniBatch = batch.slice(offset, length)
            val input = miniBatch.getInput()
            val target = miniBatch.getTarget()
            val output = workingModels(b).forward(input)
            val validatMethods = localVMethods(b).get
            validatMethods.map(validation => {
              validation(output, target)
            })
          }.reduce((left, right) => {
            left.zip(right).map { case (l, r) =>
              l + r
            }
          })
        })
      }).reduce((left, right) => {
        left.zip(right).map { case (l, r) =>
          l + r
        }
      }).zip(vMethods)
    results.foreach(r => {
      // TODO:
      DistriOptimizer.logger.info(s"${r._2} is ${r._1}")
    })
    if (validationSummary.isDefined && step > 0) {
      results.foreach { r =>
        val result = r._1.result
        validationSummary.get.addScalar(r._2.toString(), result._1,
          step - 1
        )
      }
    }
    results.map(a => (a._2, a._1)).toMap
  }

  protected def getLatestFile(path: String, fileName: String): String = {
    val fl = new java.io.File(path)
    val files = fl.listFiles(new FilenameFilter {
      override def accept(dir: File, name: String): Boolean = {
        name.startsWith(fileName)
      }
    })

    var lastMod = Long.MinValue
    var choice: String = null
    files.map {file =>
      if (file.lastModified() > lastMod) {
        choice = file.getPath;
        lastMod = file.lastModified();
      }
    }
    return choice;
  }

  def unpersistCachedModel[T: ClassTag](
      models: RDD[DistriOptimizer.Cache[T]] ): Unit = {
    models.mapPartitions { iter =>
      iter.foreach { arrayModels =>
        arrayModels.localModels.foreach(_.release())
      }
      iter
    }.count()
    models.unpersist()
  }
}
