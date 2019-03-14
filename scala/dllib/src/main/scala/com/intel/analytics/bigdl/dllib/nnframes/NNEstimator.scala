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

package com.intel.analytics.zoo.pipeline.nnframes

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}

import com.intel.analytics.bigdl.dataset.{SampleToMiniBatch, _}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Tensor, DoubleType => TensorDouble, FloatType => TensorFloat}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.zoo.common.CheckedObjectInputStream
import com.intel.analytics.zoo.feature.common.{Preprocessing, _}
import com.intel.analytics.zoo.pipeline.api.Net
import com.intel.analytics.zoo.pipeline.api.keras.layers.utils.EngineRef
import org.apache.hadoop.fs.Path
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.ml.adapter.{HasFeaturesCol, HasPredictionCol, SchemaUtils}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util._
import org.apache.spark.ml.{DLEstimatorBase, DLTransformerBase, DefaultParamsWriterWrapper, VectorCompatibility}
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.json4s.JsonDSL._
import org.json4s.{DefaultFormats, JObject}

import scala.reflect.ClassTag

private[nnframes] trait HasBatchSize extends Params {

  /**
   * Global batch size across the cluster.
   */
  final val batchSize: IntParam = new IntParam(this, "batchSize", "batchSize")

  def getBatchSize: Int = $(batchSize)
}

private[nnframes] trait TrainingParams[@specialized(Float, Double) T] extends Params {

  /**
   * When to stop the training, passed in a [[Trigger]]. E.g. Trigger.maxIterations
   */
  final val endWhen = new Param[Trigger](this, "endWhen", "Trigger to stop the training")

  def getEndWhen: Trigger = $(endWhen)

  /**
   * learning rate for the optimizer in the NNEstimator.
   * Default: 0.001
   * :: deprecated, please set the learning rate with optimMethod directly.
   */
  @deprecated("Please set the learning rate with optimMethod directly", "0.4.0")
  final val learningRate = new DoubleParam(
    this, "learningRate", "learningRate", ParamValidators.gt(0))

  def getLearningRate: Double = $(learningRate)

  /**
   * learning rate decay for each iteration.
   * Default: 0
   * :: deprecated, please set the learning rate decay with optimMethod directly.
   */
  @deprecated("Please set the learning rate decay with optimMethod directly", "0.4.0")
  final val learningRateDecay = new DoubleParam(this, "learningRateDecay", "learningRateDecay")

  def getLearningRateDecay: Double = $(learningRateDecay)

  /**
   * Number of max Epoch for the training, an epoch refers to a traverse over the training data
   * Default: 50
   */
  final val maxEpoch = new IntParam(this, "maxEpoch", "number of max Epoch", ParamValidators.gt(0))

  def getMaxEpoch: Int = $(maxEpoch)

  /**
   * optimization method to be used. BigDL supports many optimization methods like Adam,
   * SGD and LBFGS. Refer to package com.intel.analytics.bigdl.optim for all the options.
   * Default: SGD
   */
  final val optimMethod = new Param[OptimMethod[T]](this, "optimMethod", "optimMethod")

  def getOptimMethod: OptimMethod[T] = $(optimMethod)

  /**
   * Constant gradient clipping thresholds.
   */
  final val constantGradientClippingParams = new Param[(Float, Float)](this,
    "threshold for constant clipping", "constantGradientClippingParams")

  /**
   * L2 norm gradient clipping threshold.
   */
  final val l2GradientClippingParams = new FloatParam(this,
    "threshold for l2 norm gradient clipping", "l2GradientClippingParams")

  /**
   * whether to cache the Samples after preprocessing.
   * Default: true
   */
  final val cachingSample = new BooleanParam(
    this, "cachingSample", "whether to cache the Samples after preprocessing")

  def isCachingSample: Boolean = $(cachingSample)

  /**
   * Set a check point saved at `path` triggered by `trigger`
   * Default: not enabled
   */
  final val checkpointPath = new Param[String](this, "checkpointPath", "path for check points")
  final val checkpointTrigger = new Param[Trigger](this, "checkpointTrigger",
    "Trigger for check points")
  final val checkpointOverwrite = new BooleanParam(this, "checkpointOverwrite",
    "checkpointOverwrite")

  /**
   * Get check point path.
   */
  def getCheckpointPath: String = $(checkpointPath)
}

/**
 * Common trait for NNEstimator and NNModel
 */
private[nnframes] trait NNParams[@specialized(Float, Double) T] extends HasFeaturesCol
  with HasPredictionCol with HasBatchSize with VectorCompatibility {

  final val samplePreprocessing = new Param[Preprocessing[Any, Sample[T]]](this,
    "samplePreprocessing", "samplePreprocessing ")

  def getSamplePreprocessing: Preprocessing[Any, Sample[T]] = $(samplePreprocessing)

  protected def unwrapVectorAsNecessary(colType: DataType): (Row, Int) => Any = {
    // to support both ML Vector and MLlib Vector
    if (colType.typeName.contains("vector")) {
      (row: Row, index: Int) => getVectorSeq(row, colType, index)
    } else {
      (row: Row, index: Int) => row.get(index)
    }
  }
  // set default here to apply to both estimator and model
  setDefault(batchSize -> 1)
}

/**
 * [[NNEstimator]] extends [[org.apache.spark.ml.Estimator]] and supports training a BigDL
 * model with Spark DataFrame data. It can be integrated into a standard Spark ML Pipeline
 * to allow users combine the components of BigDL and Spark MLlib.
 *
 * [[NNEstimator]] supports different feature and label data type through [[Preprocessing]]. We
 * provide pre-defined [[Preprocessing]] for popular data types like Array or Vector in package
 * [[com.intel.analytics.zoo.feature]], while user can also develop customized [[Preprocessing]].
 * During fit, NNEstimator will extract feature and label data from input DataFrame and use
 * the [[Preprocessing]] to prepare data for the model. Using the [[Preprocessing]] allows
 * [[NNEstimator]] to cache only the raw data and decrease the memory consumption during feature
 * conversion and training.
 * More concrete examples are available in package [[com.intel.analytics.zoo.examples.nnframes]]
 *
 * @param model BigDL module to be optimized
 * @param criterion BigDL criterion
 * @tparam T data type of BigDL Model
 */
class NNEstimator[T: ClassTag] private[zoo] (
    @transient val model: Module[T],
    val criterion : Criterion[T],
    override val uid: String = Identifiable.randomUID("nnestimator")
  )(implicit ev: TensorNumeric[T])
  extends DLEstimatorBase[NNEstimator[T], NNModel[T]] with NNParams[T]
    with TrainingParams[T] {

  def setSamplePreprocessing[FF <: Any, LL <: Any](
      value: Preprocessing[(FF, Option[LL]), Sample[T]]): this.type =
    set(samplePreprocessing, value.asInstanceOf[Preprocessing[Any, Sample[T]]])

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /**
   * Set global batch size across the cluster. Global batch size = Batch per thread * num of cores.
   */
  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def setEndWhen(trigger: Trigger): this.type = set(endWhen, trigger)

  /**
   * :: deprecated, please set the learning rate with optimMethod directly.
   */
  @deprecated("Please set with optimMethod directly", "0.4.0")
  def setLearningRate(value: Double): this.type = set(learningRate, value)
  setDefault(learningRate -> 1e-3)

  /**
   * :: deprecated, please set with optimMethod directly.
   */
  @deprecated("Please set with optimMethod directly.", "0.4.0")
  def setLearningRateDecay(value: Double): this.type = set(learningRateDecay, value)
  setDefault(learningRateDecay -> 0.0)

  def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)
  setDefault(maxEpoch -> 50)

  def setOptimMethod(value: OptimMethod[T]): this.type = set(optimMethod, value)
  set(optimMethod, new SGD[T])

  def setConstantGradientClipping(min: Float, max: Float): this.type = {
    set(constantGradientClippingParams, (min, max))
  }

  def setGradientClippingByL2Norm(clipNorm: Float): this.type = {
    set(l2GradientClippingParams, clipNorm)
  }

  def setCachingSample(value: Boolean): this.type = {
    set(cachingSample, value)
  }
  setDefault(cachingSample, true)

  /**
   * Clear clipping params, in this case, clipping will not be applied.
   */
  def clearGradientClipping(): this.type = {
    clear(l2GradientClippingParams)
    clear(constantGradientClippingParams)
  }

  @transient private var trainSummary: Option[TrainSummary] = None

  def getTrainSummary: Option[TrainSummary] = trainSummary

  /**
   * Statistics (LearningRate, Loss, Throughput, Parameters) collected during training for the
   * training data, which can be used for visualization via Tensorboard.
   * Use setTrainSummary to enable train logger. Then the log will be saved to
   * logDir/appName/train as specified by the parameters of TrainSummary.
   *
   * Default: Not enabled
   */
  def setTrainSummary(value: TrainSummary): this.type = {
    this.trainSummary = Some(value)
    this
  }

  @transient private var validationSummary: Option[ValidationSummary] = None

  /**
   * Statistics (LearningRate, Loss, Throughput, Parameters) collected during training for the
   * validation data if validation data is set, which can be used for visualization via
   * Tensorboard. Use setValidationSummary to enable validation logger. Then the log will be
   * saved to logDir/appName/ as specified by the parameters of validationSummary.
   *
   * Default: None
   */
  def getValidationSummary: Option[ValidationSummary] = validationSummary

  /**
   * Enable validation Summary. Default: not enabled.
   */
  def setValidationSummary(value: ValidationSummary): this.type = {
    this.validationSummary = Some(value)
    this
  }

  @transient protected var validationTrigger: Option[Trigger] = None
  @transient protected var validationDF: DataFrame = _
  @transient protected var validationMethods: Array[ValidationMethod[T]] = _
  @transient protected var validationBatchSize: Int = 0
  /**
   * Set a validate evaluation during training. Default: not enabled.
   *
   * @param trigger how often to evaluation validation set
   * @param validationDF validate data set
   * @param vMethods a set of validation method [[ValidationMethod]]
   * @param batchSize batch size for validation
   * @return this estimator
   */
  def setValidation(trigger: Trigger, validationDF: DataFrame,
                    vMethods : Array[ValidationMethod[T]], batchSize: Int)
  : this.type = {
    this.validationTrigger = Some(trigger)
    this.validationDF = validationDF
    this.validationMethods = vMethods
    this.validationBatchSize = batchSize
    this
  }

  /**
   * get the validate configuration during training
   *
   * @return an Option of Tuple(ValidationTrigger, Validation data, Array[ValidationMethod[T] ],
   *         batchsize)
   */
  def getValidation: Option[(Trigger, DataFrame, Array[ValidationMethod[T]], Int)] = {
    if (validationTrigger.isDefined) {
      Some(validationTrigger.get, validationDF, validationMethods, validationBatchSize)
    }
    else {
      None
    }
  }

  /**
   * Set check points during training. Not enabled by default.
   *
   * @param path the directory to save
   * @param trigger how often to save the check point
   * @param isOverWrite: whether to overwrite existing snapshots in path. Default is True
   * @return this estimator
   */
  def setCheckpoint(path: String, trigger: Trigger, isOverWrite: Boolean = true): this.type = {
    require(path != null && trigger != null, "checkpoint path and trigger cannot be null")
    set(checkpointPath, path)
    set(checkpointTrigger, trigger)
    set(checkpointOverwrite, isOverWrite)
    this
  }

  protected def validateParams(schema : StructType): Unit = {
    if (isSet(endWhen) && isSet(maxEpoch)) {
      throw new IllegalArgumentException(s"endWhen and maxEpoch cannot be both set")
    }
    if (validationTrigger.isEmpty && validationSummary.isDefined) {
      throw new IllegalArgumentException(
        s"validationSummary is only valid if validation data is set.")
    }
  }

  override def transformSchema(schema : StructType): StructType = {
    validateParams(schema)
    ev.getType() match {
      case TensorDouble =>
        SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, false))
      case TensorFloat =>
        SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(FloatType, false))
      case _ => throw new Exception("Only support Double and Float for now")
    }
  }

  private def getDataSet(
      dataFrame: DataFrame,
      batchSize: Int): DataSet[MiniBatch[T]] = {

    val sp = $(samplePreprocessing).asInstanceOf[Preprocessing[(Any, Option[Any]), Sample[T]]]
    val featureColIndex = dataFrame.schema.fieldIndex($(featuresCol))
    val featureType = dataFrame.schema($(featuresCol)).dataType
    val featureFunc = unwrapVectorAsNecessary(featureType)

    val labelFunc: (Row) => Option[Any] = if (dataFrame.columns.contains($(labelCol))) {
      val lci = dataFrame.schema.fieldIndex($(labelCol))
      val labelFunc = unwrapVectorAsNecessary(dataFrame.schema($(labelCol)).dataType)
      (row: Row) => Some(labelFunc(row, lci))
    } else {
      (row: Row) => None
    }

    val featureAndLabel = dataFrame.rdd.map { row =>
      val features = featureFunc(row, featureColIndex)
      val labels = labelFunc(row)
      (features, labels)
    }
    val initialDataSet = if ($(cachingSample)) {
      DataSet.rdd(sp.apply(featureAndLabel))
    } else {
      DataSet.rdd(featureAndLabel).transform(sp)
    }

    initialDataSet.transform(SampleToMiniBatch[T](batchSize))
  }

  protected override def internalFit(dataFrame: DataFrame): NNModel[T] = {
    val trainingDataSet = getDataSet(dataFrame, $(batchSize))
    val endTrigger = if (isSet(endWhen)) $(endWhen) else Trigger.maxEpoch($(maxEpoch))
    val optimizer = Optimizer(model, trainingDataSet, criterion)
      .setOptimMethod($(optimMethod))
      .setEndWhen(endTrigger)

    // only set learning rate if user specifically set the values, otherwise use the
    // learning rate from $(optimMethod)
    if (isSet(learningRate) || isSet(learningRateDecay)) {
      val state = T()
      if (isSet(learningRate)) {
        state.add(T("learningRate" -> $(learningRate)))
      }
      if (isSet(learningRateDecay)) {
        state.add(T("learningRateDecay" -> $(learningRateDecay)))
      }
      optimizer.setState(state)
    }

    if (isSet(l2GradientClippingParams)) {
      optimizer.setGradientClippingByl2Norm($(l2GradientClippingParams))
    }

    if (isSet(constantGradientClippingParams)) {
      val constantClippingValues = $(constantGradientClippingParams)
      optimizer.setConstantGradientClipping(constantClippingValues._1, constantClippingValues._2)
    }

    if (validationTrigger.isDefined) {
      val validationSamples = getDataSet(validationDF, validationBatchSize)
      optimizer.setValidation(
        validationTrigger.get,
        validationSamples,
        validationMethods)
      if (this.validationSummary.isDefined) {
        optimizer.setValidationSummary(this.validationSummary.get)
      }
    }

    if (this.trainSummary.isDefined) {
      optimizer.setTrainSummary(this.trainSummary.get)
    }

    if (isSet(this.checkpointPath)) {
      optimizer.setCheckpoint($(checkpointPath), $(checkpointTrigger))
      if ($(checkpointOverwrite)) {
        optimizer.overWriteCheckpoint()
      }
    }

    val optimizedModel = optimizer.optimize()
    wrapBigDLModel(optimizedModel)
  }

  /**
   * sub classes can extend the method and return required model for different transform tasks
   */
  protected def wrapBigDLModel(m: Module[T]): NNModel[T] = {
    val dlModel = new NNModel[T](m)
    copyValues(dlModel.setParent(this))
    val clonedTransformer = ToTuple() -> $(samplePreprocessing)
      .asInstanceOf[Preprocessing[(Any, Option[Any]), Sample[T]]].clonePreprocessing()
    dlModel.setSamplePreprocessing(clonedTransformer)
  }

  /**
   * Return a deep copy for DLEstimator.
   * Note that trainSummary and validationSummary will not be copied to the new instance since
   * currently they are not thread-safe.
   */
  override def copy(extra: ParamMap): NNEstimator[T] = {
    val copied = copyValues(
      new NNEstimator[T](
        model.cloneModule(),
        criterion.cloneCriterion(),
        this.uid
      ), extra)

    if (this.validationTrigger.isDefined) {
      copied.setValidation(
        validationTrigger.get, validationDF, validationMethods.clone(), validationBatchSize)
    }
    copied
  }
}

object NNEstimator {

  /**
   * Construct a [[NNEstimator]] with default Preprocessing: A SeqToTensor
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   */
  def apply[T: ClassTag](
      model: Module[T],
      criterion: Criterion[T]
    )(implicit ev: TensorNumeric[T]): NNEstimator[T] = {
    new NNEstimator(model, criterion)
      .setSamplePreprocessing(FeatureLabelPreprocessing(SeqToTensor(), SeqToTensor()))
  }

  /**
   * Construct a [[NNEstimator]] with a feature size and label size. The constructor is useful
   * when the feature column and label column contains the following data types:
   * Float, Double, Int, Array[Float], Array[Double], Array[Int] and MLlib Vector. The feature and
   * label data are converted to Tensors with the specified sizes before sending to the model.
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
   *                    width * height = 28 * 28, featureSize = Array(28, 28).
   * @param labelSize The size (Tensor dimensions) of the label data.
   */
  def apply[T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featureSize : Array[Int],
      labelSize : Array[Int]
    )(implicit ev: TensorNumeric[T]): NNEstimator[T] = {
    new NNEstimator(model, criterion)
      .setSamplePreprocessing(FeatureLabelPreprocessing(
        SeqToTensor(featureSize), SeqToTensor(labelSize))
    )
  }

  /**
   * Construct a [[NNEstimator]] with a feature Preprocessing and label Preprocessing.
   *
   * @param model BigDL module to be optimized
   * @param criterion BigDL criterion method
   * @param featurePreprocessing Preprocessing[Any, Tensor[T] ]
   * @param labelPreprocessing Preprocessing[Any, Tensor[T] ]
   */
  def apply[F, L, T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featurePreprocessing: Preprocessing[F, Tensor[T]],
      labelPreprocessing: Preprocessing[L, Tensor[T]]
    )(implicit ev: TensorNumeric[T]): NNEstimator[T] = {
    new NNEstimator(model, criterion)
      .setSamplePreprocessing(FeatureLabelPreprocessing(featurePreprocessing, labelPreprocessing))
  }

  /**
   * Construct a [[NNEstimator]] with a featurePreprocessing only. The constructor is useful
   * when both feature and label are derived from the same column of the original DataFrame.
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   * @param featurePreprocessing A [[Preprocessing]] that transforms the feature data to a
   *        Sample[T].
   */
  def apply[F, T: ClassTag](
      model: Module[T],
      criterion: Criterion[T],
      featurePreprocessing: Preprocessing[F, Sample[T]]
    )(implicit ev: TensorNumeric[T]): NNEstimator[T] = {
    new NNEstimator(model, criterion)
      .setSamplePreprocessing(TupleToFeatureAdapter(featurePreprocessing))
  }
}

/**
 * [[NNModel]] extends Spark ML Transformer and supports BigDL model with Spark DataFrame data.
 *
 * [[NNModel]] supports different feature data type through [[Preprocessing]]. We
 * provide pre-defined [[Preprocessing]] for popular data types like Array or Vector in package
 * [[com.intel.analytics.zoo.feature]], while user can also develop
 * customized [[Preprocessing]].
 * During transform, [[NNModel]] will extract feature data from input DataFrame and use
 * the [[Preprocessing]] to prepare data for the model.
 *
 * After transform, the prediction column contains the output of the model as Array[T], where
 * T (Double or Float) is decided by the model type.
 *
 * @param model trained BigDL models to use in prediction.
 */
class NNModel[T: ClassTag] private[zoo] (
    @transient val model: Module[T],
    override val uid: String = "DLModel")(implicit ev: TensorNumeric[T])
  extends DLTransformerBase[NNModel[T]] with NNParams[T]
    with HasBatchSize with MLWritable {

  @transient
  private val logger = Logger.getLogger(getClass)

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /**
   * Set global batch size across the cluster. Global batch size = Batch per thread * num of cores.
   */
  def setBatchSize(value: Int): this.type = set(batchSize, value)

  /**
   * set Preprocessing.
   * @param value: A [[Preprocessing]] that transforms the feature data to a Sample[T].
   */
  def setSamplePreprocessing[FF <: Any](value: Preprocessing[FF, Sample[T]]): this.type =
    set(samplePreprocessing, value.asInstanceOf[Preprocessing[Any, Sample[T]]])

  /**
   * Perform a prediction on featureCol, and write result to the predictionCol.
   */
  protected override def internalTransform(dataFrame: DataFrame): DataFrame = {

    val featureColIndex = dataFrame.schema.fieldIndex($(featuresCol))
    val featureType = dataFrame.schema($(featuresCol)).dataType
    val featureFunc = unwrapVectorAsNecessary(featureType)

    val sc = dataFrame.sqlContext.sparkContext
    val modelBroadCast = ModelBroadcast[T]().broadcast(sc, model.evaluate())
    // note that here we use batch per thread, but not batch per partition. For inference,
    // GlobalBatchSize = batchPerThread * coreNumber() appears to be more intuitive for the users
    val totalNumCores = EngineRef.getCoreNumber() * EngineRef.getNodeNumber()
    val batchPerThread = Math.ceil($(batchSize).toDouble / totalNumCores).toInt
    if ($(batchSize) % totalNumCores != 0) {
      logger.warn(s"Global batch size (${$(batchSize)}) cannot be divided by total core number" +
        s"($totalNumCores). Setting batch per thread as ($batchPerThread), and actual Global" +
        s" batch size is updated to ${totalNumCores * batchPerThread}")
    } else {
      logger.info(s"Batch per thread: $batchPerThread; Total number of cores: $totalNumCores;" +
        s" Global batch size: ${batchPerThread * totalNumCores}")
    }

    val featureTransformersBC = sc.broadcast($(samplePreprocessing))
    val toBatchBC = sc.broadcast(SampleToMiniBatch[T](batchPerThread, partitionNum = Some(1)))

    // concat the prediction and other columns in DF. avoid zip between RDD
    val resultRDD = dataFrame.rdd.mapPartitions { rowIter =>
      val localModel = modelBroadCast.value()
      val featureSteps = featureTransformersBC.value.cloneTransformer()
      val toBatch = toBatchBC.value.cloneTransformer()

      rowIter.grouped(batchPerThread).flatMap { rowBatch =>
        val featureSeq = rowBatch.map(r => featureFunc(r, featureColIndex))
        val samples = featureSteps(featureSeq.iterator)
        val predictions = toBatch(samples).flatMap { batch =>
          val batchResult = localModel.forward(batch.getInput()).toTensor
          if (batchResult.size().length == 2) {
            batchResult.split(1).map(outputToPrediction)
          } else if (batchResult.size().length == 1) {
            Array(outputToPrediction(batchResult))
          } else {
            throw new RuntimeException(
              "unexpected batchResult dimension: " + batchResult.size().mkString(", "))
          }
        }
        rowBatch.toIterator.zip(predictions).map { case (row, predict) =>
          Row.fromSeq(row.toSeq ++ Seq(predict))
        }
      }
    }

    val resultSchema = transformSchema(dataFrame.schema)
    dataFrame.sqlContext.createDataFrame(resultRDD, resultSchema)
  }

  protected def outputToPrediction(output: Tensor[T]): Any = {
    output.clone().storage().array()
  }

  override def transformSchema(schema : StructType): StructType = {
    ev.getType() match {
      case TensorDouble =>
        SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, false))
      case TensorFloat =>
        SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(FloatType, false))
      case _ => throw new Exception("Only support Double and Float for now")
    }
  }

  override def copy(extra: ParamMap): NNModel[T] = {
    val copied = new NNModel[T](model.cloneModule(), uid).setParent(parent)
    copyValues(copied, extra)
  }

  override def write: MLWriter = new NNModel.NNModelWriter[T](this)
}

object NNModel extends MLReadable[NNModel[_]] {
  /**
   * Construct a [[NNModel]] with default Preprocessing: SeqToTensor
   *
   * @param model BigDL module to be optimized
   */
  def apply[T: ClassTag](
      model: Module[T]
    )(implicit ev: TensorNumeric[T]): NNModel[T] = {
    new NNModel(model)
      .setSamplePreprocessing(SeqToTensor() -> TensorToSample())
  }

  /**
   * Construct a [[NNModel]] with a feature size.
   *
   * @param model BigDL module to be optimized
   * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
   *                    width * height = 28 * 28, featureSize = Array(28, 28).
   */
  def apply[T: ClassTag](
      model: Module[T],
      featureSize: Array[Int]
    )(implicit ev: TensorNumeric[T]): NNModel[T] = {
    new NNModel(model)
      .setSamplePreprocessing(SeqToTensor(featureSize) -> TensorToSample())
  }

  /**
   * Construct a [[NNModel]] with a feature Preprocessing.
   *
   * @param model BigDL module to be optimized
   * @param featurePreprocessing Preprocessing[F, Tensor[T] ].
   */
  def apply[F, T: ClassTag](
      model: Module[T],
      featurePreprocessing: Preprocessing[F, Tensor[T]]
    )(implicit ev: TensorNumeric[T]): NNModel[T] = {
    new NNModel(model).setSamplePreprocessing(featurePreprocessing -> TensorToSample())
  }

  import scala.language.existentials
  implicit val format: DefaultFormats.type = DefaultFormats

  private[nnframes] class NNModelReader() extends MLReader[NNModel[_]] {
    override def load(path: String): NNModel[_] = {
      val (meta, model, typeTag, feaTran) = NNModel.getMetaAndModel(path, sc)
      val featureSize = (meta.metadata \ "featureSize").extract[Seq[Int]].toArray
      val nnModel = typeTag match {
        case "TensorDouble" =>
          new NNModel[Double](model.asInstanceOf[Module[Double]])
            .setSamplePreprocessing(feaTran.asInstanceOf[Preprocessing[Any, Sample[Double]]])
        case "TensorFloat" =>
          new NNModel[Float](model.asInstanceOf[Module[Float]])
            .setSamplePreprocessing(feaTran.asInstanceOf[Preprocessing[Any, Sample[Float]]])
        case _ =>
          throw new Exception("Only support float and double for now")
      }

      DefaultParamsWriterWrapper.getAndSetParams(nnModel, meta)
      nnModel
    }
  }

  private[nnframes] def getMetaAndModel(path: String, sc: SparkContext) = {
    Net // this is necessary to load Net and register the serializer
    val meta = DefaultParamsWriterWrapper.loadMetadata(path, sc)
    val (modulePath, weightPath) =
      new Path(path, "module").toString -> new Path(path, "weight").toString
    val typeTag = (meta.metadata \ "tensorDataType").extract[String]
    val model = typeTag match {
      case "TensorDouble" =>
        ModuleLoader.loadFromFile[Double](modulePath, weightPath)
      case "TensorFloat" =>
        ModuleLoader.loadFromFile[Float](modulePath, weightPath)
      case _ =>
        throw new Exception("Only support float and double for now")
    }

    val ois = new CheckedObjectInputStream(classOf[Preprocessing[Any, Any]],
      new FileInputStream(new Path(path, "samplePreprocessing").toString))
    val featurePreprocessing = try {
      ois.readObject.asInstanceOf[Preprocessing[Any, Any]]
    } finally {
      ois.close()
    }

    (meta, model, typeTag, featurePreprocessing)
  }

  class NNModelWriter[@specialized(Float, Double) T: ClassTag](
    instance: NNModel[T])(implicit ev: TensorNumeric[T]) extends MLWriter {
    override protected def saveImpl(path: String): Unit = {
      NNModel.saveImpl[T](instance, instance.model,
        path, sc, shouldOverwrite)
    }
  }

  /**
   * Helper method for saving a NNModel to disk.
   * For compatibility with spark ml pipeline, TensorDataType is stored separately in extraMetadata.
   *
   * @tparam T TensorDataType
   * @param instance  NNModel
   * @param path  Path to which to save the NNModel.
   * @param extraMetadata  Metadata such as featureSize.
   */
  private[nnframes] def saveImpl[@specialized(Float, Double) T: ClassTag](
      instance: NNModel[T],
      module: Module[T],
      path: String,
      sc: SparkContext,
      isOverWrite: Boolean = false,
      extraMetadata: Option[JObject] = None)(implicit ev: TensorNumeric[T]): Unit = {
    val tensorDataType = ev.getType() match {
      case TensorDouble => "TensorDouble"
      case TensorFloat => "TensorFloat"
      case _ => throw new Exception("Only support Double and Float for now")
    }

    val extra = extraMetadata.getOrElse(JObject()) ~ ("tensorDataType" -> tensorDataType)
    // bypass the default save for samplePreprocessing
    val spCache = instance.getSamplePreprocessing
    instance.clear(instance.samplePreprocessing)
    DefaultParamsWriterWrapper.saveMetadata(instance, path, sc, Option(extra))
    instance.setSamplePreprocessing(spCache)
    val (modulePath, weightPath) =
      new Path(path, "module").toString -> new Path(path, "weight").toString
    module.saveModule(modulePath, weightPath, isOverWrite)
    val fos = new FileOutputStream(new Path(path, "samplePreprocessing").toString)
    val oos = new ObjectOutputStream(fos)
    try {
      oos.writeObject(spCache)
    } finally {
      oos.close()
    }
  }

  override def read: MLReader[NNModel[_]] = new NNModelReader

  override def load(path: String): NNModel[_] = read.load(path)
}
