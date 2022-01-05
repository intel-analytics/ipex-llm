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

package com.intel.analytics.bigdl.dllib.keras.models

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
import com.intel.analytics.bigdl.dllib.feature._
import com.intel.analytics.bigdl.dllib.feature.image.ImageSet
import com.intel.analytics.bigdl.dllib.feature.text._
import com.intel.analytics.bigdl.dllib.keras.{Net, Predictable}
import com.intel.analytics.bigdl.dllib.keras.autograd.{Lambda, Variable}
import com.intel.analytics.bigdl.dllib.keras.autograd._
import com.intel.analytics.bigdl.dllib.keras.layers.Input
import com.intel.analytics.bigdl.dllib.keras.layers.utils._
import com.intel.analytics.bigdl.dllib.keras.Model
import com.intel.analytics.bigdl.dllib.net.NetUtils
import com.intel.analytics.bigdl.dllib.estimator.{AbstractEstimator, ConstantClipping,
GradientClipping, L2NormClipping}
import com.intel.analytics.bigdl.dllib.feature.common._
import com.intel.analytics.bigdl.dllib.nnframes.NNImageSchema
import org.apache.commons.lang.exception.ExceptionUtils
import org.apache.commons.lang3.SerializationUtils
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.Path
import org.apache.logging.log4j.LogManager
import org.apache.spark.{SparkContext, TaskContext}
import org.apache.spark.rdd.{RDD, ZippedPartitionsWithLocalityRDD}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.VectorCompatibility
import org.apache.spark.ml.feature.VectorAssembler

import scala.collection.JavaConverters._
import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag
import scala.language.implicitConversions

abstract class KerasNet[T](implicit val tag: ClassTag[T], implicit val ev: TensorNumeric[T])
  extends KerasLayer[Activity, Activity, T] with Net with Predictable[T] with VectorCompatibility {

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
        InternalOptimizer(model = this,
          dataset = distriDataSet,
          criterion = this.criterion)
      case distriFeatureSet: DistributedFeatureSet[MiniBatch[T]] =>
        InternalOptimizer(model = this,
          dataset = distriFeatureSet.toDistributed(),
          criterion = this.criterion)
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
      case disV2: InternalDistriOptimizerV2[T] =>
        disV2.setTrainData(x)
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
      case disV2: DistriOptimizerV2[T] =>
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

  private def getDataSet(
      dataFrame: DataFrame,
      batchSize: Int,
      featureCols: Array[String],
      labelCols: Array[String],
      preprocessing: Preprocessing[(Any, Option[Any]), Sample[T]]): FeatureSet[MiniBatch[T]] = {

    val sp = FeatureLabelPreprocessing(SeqToTensor(), ScalarToTensor())
      .asInstanceOf[Preprocessing[(Any, Option[Any]), Sample[T]]]

    val guid = java.util.UUID.randomUUID.toString
    val internalFeatureCol = "features" + guid
    val internalLabelCol = "labels" + guid
    val df = if (featureCols.size > 1) {
      val assembler = new VectorAssembler()
        .setInputCols(featureCols)
        .setOutputCol(internalFeatureCol)
      assembler.transform(dataFrame)
    } else {
      dataFrame.withColumnRenamed(featureCols.head, internalFeatureCol)
    }

    val assembleDF = if (labelCols.size > 1) {
      val assembler = new VectorAssembler()
        .setInputCols(labelCols)
        .setOutputCol(internalLabelCol)
      assembler.transform(df)
    } else {
      df.withColumnRenamed(labelCols.head, internalLabelCol)
    }

    val featureColIndex = assembleDF.schema.fieldIndex(internalFeatureCol)
    val featureType = assembleDF.schema(internalFeatureCol).dataType
    val featureFunc = unwrapVectorAsNecessary(featureType)

    val labelFunc: (Row) => Option[Any] = {
      val lci = assembleDF.schema.fieldIndex(internalLabelCol)
      val labelFunc = unwrapVectorAsNecessary(assembleDF.schema(internalLabelCol).dataType)
      (row: Row) => Some(labelFunc(row, lci))
    }

    val featureAndLabel = assembleDF.rdd.map { row =>
      val features = featureFunc(row, featureColIndex)
      val labels = labelFunc(row)
      (features, labels)
    }

    val initialDataSet = FeatureSet.rdd(featureAndLabel).transform(sp)
    initialDataSet.transform(SampleToMiniBatch[T](batchSize))
  }

  def fit(
      x: DataFrame,
      batchSize: Int,
      nbEpoch: Int,
      featureCols: Array[String],
      labelCols: Array[String],
      valX: DataFrame)(implicit ev: TensorNumeric[T]): Unit = {
    val preprocessing =
      FeatureLabelPreprocessing(SeqToTensor(), ScalarToTensor())
        .asInstanceOf[Preprocessing[(Any, Option[Any]), Sample[T]]]

    val trainingData = getDataSet(x, batchSize, featureCols, labelCols,
      preprocessing).toDataSet()
    val valData = if (valX != null) {
      getDataSet(valX, batchSize, featureCols, labelCols, preprocessing).toDataSet()
    } else null

    this.fit(trainingData, nbEpoch, valData)

    predictTransformer = ToTuple() -> preprocessing
      .asInstanceOf[Preprocessing[(Any, Option[Any]), Sample[T]]].clonePreprocessing()
  }

  def fit(
      x: DataFrame,
      batchSize: Int,
      nbEpoch: Int,
      featureCols: Array[String],
      labelCols: Array[String])(implicit ev: TensorNumeric[T]): Unit = {
    this.fit(x, batchSize, nbEpoch, featureCols, labelCols, null)
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

  def toModel(): keras.Model[T]

// uncomment when migrating TFNet
//  /**
//   * Save model to keras2 h5 file. Only for inference
//   * @param filePath path to save model.
//   * @param python python path, need analytics-zoo and tensorflow installed.
//   */
//  def saveToKeras2[T: ClassTag](
//        filePath: String,
//        python: String = "python")(implicit ev: TensorNumeric[T]): Unit = {
//    Net.saveToKeras2[T](this, filePath, python)
//  }
//
//  /**
//   * Save model to tensorflow protobuf. Only for inference.
//   * @param dir directory to save model.
//   * @param python python path, need analytics-zoo and tensorflow installed.
//   */
//  def saveToTf[T: ClassTag](
//        dir: String,
//        python: String = "python")(implicit ev: TensorNumeric[T]): Unit = {
//    Net.saveToTf[T](this, dir, python)
//  }

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

object InternalOptimizer {
  def apply[T: ClassTag](
    model: Module[T],
    dataset: DistributedDataSet[MiniBatch[T]],
    criterion: Criterion[T]
  )(implicit ev: TensorNumeric[T]): Optimizer[T, MiniBatch[T]] = {
    EngineRef.getOptimizerVersion() match {
      case OptimizerV1 =>
        new InternalDistriOptimizer[T](
          _model = model,
          _dataset = dataset,
          _criterion = criterion)
      case OptimizerV2 =>
        new InternalDistriOptimizerV2[T](
          _model = model,
          _dataset = dataset,
          _criterion = criterion)
    }
  }
}

private[bigdl] object InternalOptimizerUtil {

  def setExecutorMklThread(cachedModels: RDD[_]): Unit = {
    cachedModels.mapPartitions{_ =>
      val numCores = scala.sys.env("OMP_NUM_THREADS").toInt
      System.setProperty("bigdl.mklNumThreads", numCores.toString)
      Iterator.single(1)
    }.count()
  }

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

  def endEpochV2[T: ClassTag](optimizer: DistriOptimizerV2[T]): Unit = {
    val method = classOf[DistriOptimizerV2[T]].getDeclaredMethod("endEpoch")
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
      implicit ev: TensorNumeric[T]): (RDD[DistriOptimizer.CacheV1[T]], ModelBroadcast[T]) = {
    KerasUtils.invokeMethodWithEv(DistriOptimizer,
      "com$intel$analytics$bigdl$dllib$optim$DistriOptimizer$$initThreadModels",
      args: _*).asInstanceOf[(RDD[DistriOptimizer.CacheV1[T]], ModelBroadcast[T])]
  }

  def clearState[T: ClassTag](
        models: RDD[DistriOptimizer.CacheV1[T]]): Unit = {
    KerasUtils.invokeMethod(DistriOptimizer,
      "clearState", models, implicitly[reflect.ClassTag[T]])
  }

  def clearStateV2[T: ClassTag](
        models: RDD[CacheV2[T]]): Unit = {
    KerasUtils.invokeMethod(DistriOptimizerV2,
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

// uncomment when torch net migrate
//  // TODO: Delete this when switch to Bigdl 0.11.0.
//  def getTorchModel[T: ClassTag](
//      models: RDD[CacheV1[T]],
//      parameters: AllReduceParameter[T],
//      trainingModel: TorchModel)(implicit ev: TensorNumeric[T]): TorchModel = {
//    val partitionNum = models.partitions.length
//    val extraState = models.map(_.localModels.head.getExtraParameter()).first()
//    trainingModel.setExtraParam(extraState.asInstanceOf[Array[Tensor[Float]]])
//    val (weights, gradients) = models.mapPartitions(iter => {
//      val cached = iter.next()
//      val curPartitionId = TaskContext.getPartitionId()
//      Iterator.single((Map(curPartitionId -> parameters.weightPartition),
//        Map(curPartitionId -> parameters.gradientPartition)))
//    }).reduce((a, b) => (a._1 ++ b._1, a._2 ++ b._2))
//
//    val parameterArray = trainingModel.parameters()
//    (0 until parameterArray._2.length).foreach(i =>
//      parameterArray._2(i).resizeAs(parameterArray._1(i))
//    )
//    val (parameter, gradientParameter) = getParametersFromModel(trainingModel)
//    val parameterLength = parameter.nElement()
//    val taskSize = parameterLength / partitionNum
//    require(taskSize != 0, "parameter length should not less than partition number")
//    val extraSize = parameterLength % partitionNum
//
//    (0 until partitionNum).map(pid => {
//      val start = pid * taskSize + math.min(pid, extraSize)
//      val length = taskSize + (if (pid < extraSize) 1 else 0)
//      parameter.narrow(1, start + 1, length).copy(weights(pid).asInstanceOf[Tensor[Float]])
//      gradientParameter.narrow(1, start + 1, length)
//        .copy(gradients(pid).asInstanceOf[Tensor[Float]])
//    })
//
//    trainingModel
//  }

  def releaseBroadcast[T: ClassTag](
        uuid: String)(implicit ev: TensorNumeric[T]): Unit = {
    KerasUtils.invokeMethodWithEv(
      "com.intel.analytics.bigdl.dllib.models.utils.CachedModels",
      "deleteKey",
      uuid)
  }


  def getLocalPartitionRangeFromParameters[T: ClassTag](
       parameters: AllReduceParameter[T]): (Int, Int) = {
    KerasUtils.invokeMethod(parameters, "localPartitionRange").asInstanceOf[(Int, Int)]
  }

}

private[bigdl] class InternalLocalOptimizer[T: ClassTag] (
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

private[bigdl] class InternalDistriOptimizer[T: ClassTag] (
    _model: Module[T],
    _dataset: DistributedDataSet[MiniBatch[T]],
    _criterion: Criterion[T])
  (implicit ev: TensorNumeric[T]) extends DistriOptimizer[T](_model, _dataset, _criterion)
  with AbstractEstimator[T]{
  import InternalDistriOptimizer._
  protected var checkpointDir: Option[String] = None
  protected var numSlice: Int = 1
  protected var cachedModels: RDD[DistriOptimizer.CacheV1[T]] = null
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
    state("isLayerwiseScaled") = com.intel.analytics.bigdl.dllib.nn.Utils.isLayerwiseScaled(_model)

    val nodeNumber = EngineRef.getNodeNumber()

    /**
     * The best practice of torch's training is single model in each executor.
     * And use multi OMP threads to speedup the single model's training.
     * Currently, we only provide single model + multi OMP threads for torch model.
     * TODO: support tfnet.
     */
    logger.info(s"${model} isTorch is ${model.isPyTorch()}")
    val torchOptimize = model.isPyTorch()
    val modelPerExecutor = if (torchOptimize) {
      require(EngineRef.getEngineType() != MklDnn, "torch model shouldn't use MKLDNN engine.")
      val numOmpThread = distDataset.originRDD().sparkContext
        .getConf.get("spark.executorEnv.OMP_NUM_THREADS").toInt
      logger.info(s"torch model will use ${numOmpThread} OMP threads.")
      1
    } else {
      EngineRef.getCoreNumber()
    }

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
//      if (torchOptimize) {
//        InternalOptimizerUtil.setExecutorMklThread(distDataset.originRDD())
//      }
      val modelsAndBroadcast = InternalOptimizerUtil.initThreadModels[T](
        trainingModel, distDataset, criterion, state,
        Int.box(nodeNumber), Int.box(modelPerExecutor), Boolean.box(checkSingleton),
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
          Int.box(modelPerExecutor),
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
              Module.loadModule[T](modelFile)
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
              Int.box(nodeNumber), Int.box(modelPerExecutor), Boolean.box(checkSingleton),
              allReduceParameter, parameterSplits, validationMethods, optimMethods)
            cachedModels = modelsAndBroadcast._1
            modelBroadcast = modelsAndBroadcast._2
          } else {
            throw t
          }
      }
    }

    InternalDistriOptimizer.getModel(
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

  def getTrainSummary(tag: String): Array[(Long, Float, Double)] = {
    if (this.trainSummary isDefined) {
      this.trainSummary.get.readScalar(tag)
    } else {
      null
    }
  }

  def getValidationSummary(tag: String): Array[(Long, Float, Double)] = {
    if (this.validationSummary isDefined) {
      this.validationSummary.get.readScalar(tag)
    } else {
      null
    }
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
      val validationTrigger = SerializationUtils.clone(checkPointTrigger.get)
      this.setValidation(validationTrigger, validationSet.toDataSet(), validationMethod)
    }
    if (numSlice != 1) {
      val state = InternalOptimizerUtil.getStateFromOptiMethod(
        optimMethods.values.head)
      if (checkPointTrigger.isDefined) {
        if (checkPointTrigger.get.isInstanceOf[ZooTrigger]) {
          checkPointTrigger.get.asInstanceOf[ZooTrigger].setZooState(state)
          if (validationMethod != null && validationSet != null) {
            validationTrigger.get.asInstanceOf[ZooTrigger].setZooState(state)
          }
        } else {
          throw new IllegalArgumentException(
            s"Excepted com.intel.analytics.bigdl.dllib.common.ZooTrigger." +
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

    val coresPerNode = EngineRef.getCoreNumber()
    val _subModelNumber = EngineRef.getEngineType() match {
      case MklBlas => if (_model.isPyTorch()) {
        1
      } else {
        coresPerNode
      }
      case MklDnn => 1
      case _ => throw new IllegalArgumentException
    }

    val models = if (null != cachedModels) {
      val bcVMethods = cachedModels.sparkContext.broadcast(validationMethod)
      cachedModels.map{cache =>
        CacheV1[T](
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
      val bcModel = ModelBroadcast[T]().broadcast(sc, _model)
      validateRDD.mapPartitions{_ =>
        Iterator.single(CacheV1[T](
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

private[bigdl] class InternalDistriOptimizerV2[T: ClassTag] (
    _model: Module[T],
    _dataset: DistributedDataSet[MiniBatch[T]],
    _criterion: Criterion[T])
  (implicit ev: TensorNumeric[T]) extends DistriOptimizerV2[T](_model, _dataset, _criterion)
  with AbstractEstimator[T]{
  import InternalDistriOptimizerV2._
  protected var checkpointDir: Option[String] = None
  protected var numSlice: Int = 1
  protected var cachedModels: RDD[CacheV2[T]] = null
  protected var modelBroadcast: ModelBroadcast[T] = null
  protected var parameterSplits: Map[String, (Int, Int)] = null
  protected var allReduceParameter: AllReduceParameter[T] = null

  override def close(): Unit = {
    if (cachedModels != null) {
      InternalOptimizerUtil.clearStateV2(cachedModels)
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
    InternalOptimizerUtil.endEpochV2(this)
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
      val validationTrigger = SerializationUtils.clone(checkPointTrigger.get)
      this.setValidation(validationTrigger, validationSet.toDataSet(), validationMethod)
    }
    if (numSlice != 1) {
      val state = InternalOptimizerUtil.getStateFromOptiMethod(
        optimMethods.values.head)
      if (checkPointTrigger.isDefined) {
        if (checkPointTrigger.get.isInstanceOf[ZooTrigger]) {
          checkPointTrigger.get.asInstanceOf[ZooTrigger].setZooState(state)
        } else {
          throw new IllegalArgumentException(
            s"Excepted com.intel.analytics.bigdl.dllib.common.ZooTrigger." +
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
        this.optimize()
        InternalOptimizerUtil.endEpochV2(this)
        // (neval - 1) is true iteration
        state("epoch") = Math.floor(state[Int]("currentSlice").toDouble / numSlice).toInt + 1
      }
    } else {
      this.optimize()
    }
    this
  }

  override def evaluate(
        validationSet: FeatureSet[MiniBatch[T]],
        validationMethod: Array[ValidationMethod[T]]
      ): Map[ValidationMethod[T], ValidationResult] = {
    val validateRDD = validationSet.toDistributed().data(train = false)

    val models = this.cachedModels

    // get current iteration from optimMethod
    val step = if (null != optimMethods && optimMethods.size != 0) {
      val state = InternalOptimizerUtil.getStateFromOptiMethod(
        optimMethods.values.head)
      state.getOrElse[Int]("neval", -1)
    } else {
      -1
    }

    InternalDistriOptimizerV2.validate(
      validationSet,
      validationMethod,
      models,
      step,
      validationSummary
    )
  }
}

object InternalDistriOptimizer {
  val logger = LogManager.getLogger(this.getClass)

  protected def validate[T](validationFeatureSet: FeatureSet[MiniBatch[T]],
                            validationMethods: Array[ValidationMethod[T]],
                            models: RDD[CacheV1[T]],
                            step: Int,
                            validationSummary: Option[ValidationSummary]
                           ): Map[ValidationMethod[T], ValidationResult] = {
    val vMethods = validationMethods
    val validateRDD = validationFeatureSet.toDistributed().data(train = false)

    // TODO: evaluate local
    val results = ZippedPartitionsWithLocalityRDD(models, validateRDD)(
      (modelIter, dataIter) => {
        val cached = modelIter.next()
        val workingModels = cached.localModels
        val localVMethods = cached.localMethods

        workingModels.foreach(_.evaluate())
        val _subModelNumber = workingModels.length
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
      models: RDD[DistriOptimizer.CacheV1[T]] ): Unit = {
    models.mapPartitions { iter =>
      iter.foreach { arrayModels =>
        arrayModels.localModels.foreach(_.release())
      }
      iter
    }.count()
    models.unpersist()
  }

  def getModel[T: ClassTag](models: RDD[CacheV1[T]],
                            parameters: AllReduceParameter[T],
                            trainingModel: Module[T])(implicit ev: TensorNumeric[T])
  : Module[T] = {

    if (trainingModel.isTensorFlow()) {

      // We have a special treatment here for TFTraingingHelperV2, which currently is the
      // only class that isTensorFlow() evaluates to true.
      // TFTrainingHelperV2 uses AllReduceParameter to sync gradient, not weights, so we cannot
      // get weight partitions from it. We need to get the weights directly from model.
      // The different code section is commented below.
      val partitionNum = models.partitions.length
      models.mapPartitions(iter => {
        iter.next().localModels.head.beforeGetModel()
        Iterator.single(1)
      }).reduce(_ + _)

      val extraParamLength = models.map(_.localModels.head.getExtraParameter().length).first()
      val extraState = new Array[Tensor[T]](extraParamLength)
      (0 until extraParamLength).foreach(i =>
        extraState(i) = models.map(_.localModels.head.getExtraParameter()(i)).first()
      )
      trainingModel.setExtraParameter(extraState)

      // make sure gradient is as the same length as weight
      val parameterArray = trainingModel.parameters()
      (0 until parameterArray._2.length).foreach(i =>
        parameterArray._2(i).resizeAs(parameterArray._1(i))
      )

      val (parameter, gradientParameter) =
        InternalOptimizerUtil.getParametersFromModel(trainingModel)

      val (weights, gradients) = models.mapPartitions(iter => {
        val cached = iter.next()
        val curPartitionId = TaskContext.getPartitionId()
        // different code section from regular getModel
        // section start
        val (offset, size) =
          InternalOptimizerUtil.getLocalPartitionRangeFromParameters(parameters)
        val weightTensor = Tensor[T](size)
        weightTensor.copy(cached.modelWeights.head.narrow(1, offset, size))
        Iterator.single((Map(curPartitionId -> weightTensor),
          Map(curPartitionId -> parameters.gradientPartition)))
        // section end
      }).reduce((a, b) => (a._1 ++ b._1, a._2 ++ b._2))

      val taskSize = parameters.size / partitionNum
      require(taskSize != 0, "parameter length should not less than partition number")
      val extraSize = parameters.size % partitionNum

      (0 until partitionNum).map(pid => {
        val start = parameters.paramOffset + pid * taskSize + math.min(pid, extraSize)
        val length = taskSize + (if (pid < extraSize) 1 else 0)
        parameter.narrow(1, start, length).copy(weights(pid))
        gradientParameter.narrow(1, start, length).copy(gradients(pid))
      })
    } else {
      InternalOptimizerUtil.getModel(models, parameters, trainingModel)
    }
    trainingModel
  }
}

object InternalDistriOptimizerV2 {
  val logger = LogManager.getLogger(this.getClass)

  protected def validate[T](validationFeatureSet: FeatureSet[MiniBatch[T]],
                            validationMethods: Array[ValidationMethod[T]],
                            models: RDD[CacheV2[T]],
                            step: Int,
                            validationSummary: Option[ValidationSummary]
                           ): Map[ValidationMethod[T], ValidationResult] = {
    val vMethods = validationMethods
    val validateRDD = validationFeatureSet.toDistributed().data(train = false)

    // TODO: evaluate local
    val results = ZippedPartitionsWithLocalityRDD(models, validateRDD)(
      (modelIter, dataIter) => {
        val cached = modelIter.next()
        val workingModels = cached.localModels
        val localVMethods = cached.localMethods

        workingModels.foreach(_.evaluate())
        val _subModelNumber = workingModels.length
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
      DistriOptimizerV2.logger.info(s"${r._2} is ${r._1}")
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

  def unpersistCachedModel[T: ClassTag](
      models: RDD[CacheV2[T]] ): Unit = {
      models.mapPartitions { iter =>
      iter.foreach { arrayModels =>
        arrayModels.localModels.foreach(_.release())
      }
      iter
    }.count()
    models.unpersist()
  }
}

object Models {
  def loadModel[T: ClassTag](path: String)(implicit ev: TensorNumeric[T]): keras.Model[T] = {
    val model = Net.load[T](path)
    if (!model.isInstanceOf[Model[T]]) {
      throw new RuntimeException("Not an Analytics Zoo Keras-style model.")
    }
    model.asInstanceOf[keras.Model[T]]
  }
}

object Model {
  /**
   * Build a multiple-input, multiple-output graph container.
   * @param input Array of input nodes.
   * @param output Array of output nodes.
   * @return A graph container.
   */
  def apply[T: ClassTag](
    input : Array[ModuleNode[T]],
    output : Array[ModuleNode[T]])(implicit ev: TensorNumeric[T]) : keras.Model[T] = {
    keras.Model[T](input, output)
  }

  /**
   * Build a single-input, multiple-output graph container
   * @param input The input node.
   * @param output Array of output nodes.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : Array[ModuleNode[T]])
    (implicit ev: TensorNumeric[T]) : keras.Model[T] = {
    keras.Model[T](Array(input), output)
  }

  /**
   * Build a multiple-input, single-output graph container.
   * @param input Array of input nodes.
   * @param output The output node.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : Array[ModuleNode[T]], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : keras.Model[T] = {
    keras.Model[T](input, Array(output))
  }

  /**
   * Build a single-input, single-output graph container
   * @param input The input node.
   * @param output The output node.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : ModuleNode[T], output : ModuleNode[T])
    (implicit ev: TensorNumeric[T]) : keras.Model[T] = {
    keras.Model[T](Array(input), Array(output))
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
    output : Array[Variable[T]])(implicit ev: TensorNumeric[T]) : keras.Model[T] = {
    keras.Model[T](input.map(_.node), output.map(_.node))
  }

  /**
   * Build a single-input, multiple-output graph container
   * @param input The input variable.
   * @param output Array of output variables.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : Variable[T], output : Array[Variable[T]])
    (implicit ev: TensorNumeric[T]) : keras.Model[T] = {
    keras.Model[T](Array(input.node), output.map(_.node))
  }

  /**
   * Build a multiple-input, single-output graph container.
   * @param input Array of input variables.
   * @param output The output variables.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : Array[Variable[T]], output : Variable[T])
    (implicit ev: TensorNumeric[T]) : keras.Model[T] = {
    keras.Model[T](input.map(_.node), Array(output.node))
  }

  /**
   * Build a single-input, single-output graph container
   * @param input The input variable.
   * @param output The output variable.
   * @return A graph container.
   */
  def apply[T: ClassTag](input : Variable[T], output : Variable[T])
    (implicit ev: TensorNumeric[T]) : keras.Model[T] = {
    keras.Model[T](Array(input.node), Array(output.node))
  }
}

object Sequential {
  def apply[@specialized(Float, Double) T: ClassTag]()
    (implicit ev: TensorNumeric[T]) : keras.Sequential[T] = {
    keras.Sequential[T]()
  }
}
