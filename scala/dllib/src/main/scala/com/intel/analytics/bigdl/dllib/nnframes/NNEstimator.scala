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
import com.intel.analytics.bigdl.{Criterion, DataSet, Module}
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.tensor.{Tensor, DoubleType => TensorDouble,
  FloatType => TensorFloat}
import com.intel.analytics.bigdl.utils.T
import com.intel.analytics.bigdl.utils.serializer.ModuleLoader
import com.intel.analytics.bigdl.visualization.{TrainSummary, ValidationSummary}
import com.intel.analytics.zoo.pipeline.nnframes.transformers.internal.FeatureToTupleAdapter
import com.intel.analytics.zoo.pipeline.nnframes.transformers.{ArrayToTensor,
  FeatureLabelTransformer, MLlibVectorToTensor, NumToTensor, TensorToSample}
import org.apache.spark.ml.adapter.{HasFeaturesCol, HasPredictionCol, SchemaUtils}
import org.apache.spark.ml.{DLEstimatorBase, DLTransformerBase, DefaultParamsWriterWrapper}
import org.apache.spark.ml.param._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.ml.util._
import org.apache.spark.rdd.RDD
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._

import scala.reflect.ClassTag

private[nnframes] trait HasBatchSize extends Params {

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
   */
  final val learningRate = new DoubleParam(
    this, "learningRate", "learningRate", ParamValidators.gt(0))

  def getLearningRate: Double = $(learningRate)

  /**
   * learning rate decay for each iteration.
   * Default: 0
   */
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
}

/**
 * Common trait for NNEstimator and NNModel
 */
private[nnframes] trait NNParams[F, @specialized(Float, Double) T] extends HasFeaturesCol
  with HasPredictionCol with HasBatchSize {

  setDefault(batchSize -> 1)
}

/**
 * [[NNEstimator]] extends [[org.apache.spark.ml.Estimator]] and supports training a BigDL
 * model with Spark DataFrame data. It can be integrated into a standard Spark ML Pipeline
 * to enable users for combined usage with Spark MLlib.
 *
 * [[NNEstimator]] supports different feature and label data type through transformers. We
 * provide pre-defined transformers for popular data types like Array or Vector in package
 * [[com.intel.analytics.zoo.pipeline.nnframes.transformers]], while user can also develop
 * customized Transformer which extends from [[com.intel.analytics.bigdl.dataset.Transformer]].
 * During fit, NNEstimator will extract feature and label data from input DataFrame and use
 * the transformers to prepare data for the model.
 *
 * Using the transformers allows [[NNEstimator]] to cache only the raw data and decrease
 * the memory consumption during feature conversion and training.
 *
 * More concrete examples are available in package [[com.intel.analytics.zoo.examples.nnframes]]
 *
 * Construct a NNEstimator with BigDL model, criterion and a sampleTransformer that transform a
 * (feature, label) tuple to a BigDL Sample. This constructor is only recommended for the expert
 * users. Most users should use the other constructor with featureTransformer and
 * labelTransformer.
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 * @param sampleTransformer Expert param. A transformer that transforms the (feature, label) tuple
 *        to a BigDL Sample[T], where T is decided by the BigDL model.
 *
 *        Note: sampleTransformer should be able to handle the case that label = null.
 *        During fit, [[NNEstimator]] will extract (feature, label) tuple from input
 *        DataFrame and use sampleTransformer to transform the tuple into BigDL Sample to be
 *        ingested by the model. If Label column is not available, (feature, null) will be
 *        sent to sampleTransformer.
 *
 *        The sampleTransformer will also be copied to the generated [[NNModel]] and applied
 *        to feature column during transform, where (feature, null) will be passed to the
 *        sampleTransformer.
 * @tparam F data type from feature column, E.g. Array[_] or Vector
 * @tparam L data type from label column, E.g. Float, Double, Array[_] or Vector
 * @tparam T data type of BigDL Model
 */
class NNEstimator[F, L, T: ClassTag](
    @transient val model: Module[T],
    val criterion : Criterion[T],
    val sampleTransformer: Transformer[(F, L), Sample[T]],
    override val uid: String = Identifiable.randomUID("nnEstimator")
  )(implicit ev: TensorNumeric[T])
  extends DLEstimatorBase[NNEstimator[F, L, T], NNModel[F, T]] with NNParams[F, T]
    with TrainingParams[T] {

  /**
   * Construct a NNEstimator with a featureTransformer and a labelTransformer, which convert the
   * data in feature column and label column to Tensors (Multi-dimension array) for model.
   *
   * The featureTransformer will be copied to the fitted NNModel, and apply to feature
   * column data during transform. This is the the recommended constructor for most users.
   *
   * @param model BigDL module to be optimized
   * @param criterion  BigDL criterion method
   * @param featureTransformer A transformer that transforms the feature data to a Tensor[T].
   *        Some pre-defined transformers are provided in package
   *        [[com.intel.analytics.zoo.pipeline.nnframes.transformers]]. E.g.
   *        [[ArrayToTensor]] is used to transform Array[_] in DataFrame to Tensor. For a feature
   *        column that contains 576 floats in an Array, Users can set ArrayToTensor(Array(28, 28))
   *        as featureTransformer, which will convert the feature data into Tensors with dimension
   *        28 * 28 to be processed by a convolution Model. For a simple linear model, user may
   *        just use ArrayToTensor(Array(576)), which will convert the data into Tensors with
   *        single dimension (576).
   *        [[MLlibVectorToTensor]] is used to transform [[org.apache.spark.mllib.linalg.Vector]]
   *        to a Tensor.
   *        [[NumToTensor]] transform a number to a Tensor with single dimension of length 1.
   *        Multiple transformer can be combined as a ChainedTransformer.
   * @param labelTransformer similar to featureTransformer, but applies to Label data.
   */
  def this(
      model: Module[T],
      criterion: Criterion[T],
      featureTransformer: Transformer[F, Tensor[T]],
      labelTransformer: Transformer[L, Tensor[T]]
    )(implicit ev: TensorNumeric[T]) =
    this(model, criterion, FeatureLabelTransformer(featureTransformer, labelTransformer))

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def setEndWhen(trigger: Trigger): this.type = set(endWhen, trigger)

  def setLearningRate(value: Double): this.type = set(learningRate, value)
  setDefault(learningRate -> 1e-3)

  def setLearningRateDecay(value: Double): this.type = set(learningRateDecay, value)
  setDefault(learningRateDecay -> 0.0)

  def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)
  setDefault(maxEpoch -> 50)

  def setOptimMethod(value: OptimMethod[T]): this.type = set(optimMethod, value)
  set(optimMethod, new SGD[T])

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
   * Enable validation Summary
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
   * Set a validate evaluation during training
   *
   * @param trigger how often to evaluation validation set
   * @param validationDF validate data set
   * @param vMethods a set of validation method [[ValidationMethod]]
   * @param batchSize batch size for validation
   * @return this optimizer
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

  def getValidation: Option[(Trigger, DataFrame, Array[ValidationMethod[T]], Int)] = {
    if (validationTrigger.isDefined) {
      Some(validationTrigger.get, validationDF, validationMethods, validationBatchSize)
    }
    else {
      None
    }
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

    val featureColIndex = dataFrame.schema.fieldIndex($(featuresCol))
    val labelColIndex = if (dataFrame.columns.contains($(labelCol))) {
      Some(dataFrame.schema.fieldIndex($(labelCol)))
    } else {
      None
    }

    val featureAndLabel: RDD[(F, L)] = dataFrame.rdd.map { row =>
      val features = row.getAs[F](featureColIndex)
      val labels = labelColIndex match {
        case Some(i) => row.getAs[L](i)
        case None => null.asInstanceOf[L]
      }
      (features, labels)
    }
    DataSet.rdd(featureAndLabel).transform(sampleTransformer -> SampleToMiniBatch[T](batchSize))
  }

  protected override def internalFit(dataFrame: DataFrame): NNModel[F, T] = {
    val trainingDataSet = getDataSet(dataFrame, $(batchSize))
    val state = T("learningRate" -> $(learningRate), "learningRateDecay" -> $(learningRateDecay))
    val endTrigger = if (isSet(endWhen)) $(endWhen) else Trigger.maxEpoch($(maxEpoch))
    val optimizer = Optimizer(model, trainingDataSet, criterion)
      .setState(state)
      .setOptimMethod($(optimMethod))
      .setEndWhen(endTrigger)

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

    val optimizedModel = optimizer.optimize()
    wrapBigDLModel(optimizedModel)
  }

  /**
   * sub classes can extend the method and return required model for different transform tasks
   */
  protected def wrapBigDLModel(m: Module[T]): NNModel[F, T] = {
    val clonedTransformer = FeatureToTupleAdapter(
      sampleTransformer.cloneTransformer().asInstanceOf[Transformer[(F, Any), Sample[T]]])
    val dlModel = new NNModel[F, T](m, clonedTransformer)
    copyValues(dlModel.setParent(this))
  }

  /**
   * Return a deep copy for DLEstimator.
   * Note that trainSummary and validationSummary will not be copied to the new instance since
   * currently they are not thread-safe.
   */
  override def copy(extra: ParamMap): NNEstimator[F, L, T] = {
    val copied = copyValues(
      new NNEstimator[F, L, T](
        model.cloneModule(),
        criterion.cloneCriterion(),
        sampleTransformer.cloneTransformer(),
        this.uid
      ),
      extra)

    if (this.validationTrigger.isDefined) {
      copied.setValidation(
        validationTrigger.get, validationDF, validationMethods.clone(), validationBatchSize)
    }
    copied
  }
}

/**
 * [[NNModel]] extends Spark ML Transformer and supports BigDL model with Spark DataFrame data.
 *
 * [[NNModel]] supports different feature data type through transformers. We
 * provide pre-defined transformers for popular data types like Array or Vector in package
 * [[com.intel.analytics.zoo.pipeline.nnframes.transformers]], while user can also develop
 * customized Transformer which extends from [[com.intel.analytics.bigdl.dataset.Transformer]].
 * During transform, [[NNModel]] will extract feature data from input DataFrame and use
 * the transformer to prepare data for the model.
 *
 * After transform, the prediction column contains the output of the model as Array[T], where
 * T (Double or Float) is decided by the model type.
 *
 * @param model trainned BigDL models to use in prediction.
 * @param sampleTransformer A transformer that transforms the feature data to a Sample[T].
 */
class NNModel[F, T: ClassTag](
    @transient val model: Module[T],
    @transient val sampleTransformer: Transformer[F, Sample[T]],
    override val uid: String = "DLModel")(implicit ev: TensorNumeric[T])
  extends DLTransformerBase[NNModel[F, T]] with NNParams[F, T]
    with HasBatchSize with MLWritable {

  /**
   * Construct NNModel with a BigDL model and a feature-to-tensor transformer
   * @param model trainned BigDL models to use in prediction.
   * @param featureTransformer A transformer that transforms the feature data to a Tensor[T].
   *        Some pre-defined transformers are provided in package
   *        [[com.intel.analytics.zoo.pipeline.nnframes.transformers]]. E.g.
   *        [[ArrayToTensor]] is used to transform Array[_] in DataFrame to Tensor. For a feature
   *        column that contains 576 floats in an Array, Users can set ArrayToTensor(Array(28, 28))
   *        as featureTransformer, which will convert the feature data into Tensors with dimension
   *        28 * 28 to be processed by a convolution Model. For a simple linear model, user may
   *       just use ArrayToTensor(Array(576)), which will convert the data into Tensors with
   *       single dimension (576).
   *        [[MLlibVectorToTensor]] is used to transform [[org.apache.spark.mllib.linalg.Vector]]
   *        to a Tensor.
   *        [[NumToTensor]] transform a number to a Tensor with singel dimension of length 1.
   *        Multiple transformer can be combined as a ChainedTransformer.
   */
  def this(
      model: Module[T],
      featureTransformer: Transformer[F, Tensor[T]]
    )(implicit ev: TensorNumeric[T]) =
    this(model, featureTransformer -> TensorToSample())

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  /**
   * Perform a prediction on featureCol, and write result to the predictionCol.
   */
  protected override def internalTransform(dataFrame: DataFrame): DataFrame = {

    val featureColIndex = dataFrame.schema.fieldIndex($(featuresCol))

    val sc = dataFrame.sqlContext.sparkContext
    val modelBroadCast = ModelBroadcast[T]().broadcast(sc, model.evaluate())
    val localBatchSize = $(batchSize)
    val featureTransformersBC = sc.broadcast(sampleTransformer)
    val toBatchBC = sc.broadcast(SampleToMiniBatch[T](localBatchSize))

    // concat the prediction and other columns in DF. avoid zip between RDD
    val resultRDD = dataFrame.rdd.mapPartitions { rowIter =>
      val localModel = modelBroadCast.value()
      val featureSteps = featureTransformersBC.value.cloneTransformer()
      val toBatch = toBatchBC.value.cloneTransformer()

      rowIter.grouped(localBatchSize).flatMap { rowBatch =>
        val featureSeq = rowBatch.map(r => r.getAs[F](featureColIndex))
        val samples = featureSteps(featureSeq.iterator)
        val predictions = toBatch(samples).flatMap { batch =>
          val batchResult = localModel.forward(batch.getInput()).toTensor.squeeze()
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

  override def copy(extra: ParamMap): NNModel[F, T] = {
    val copied = new NNModel[F, T](model.cloneModule(), sampleTransformer.cloneTransformer(), uid)
      .setParent(parent)
    copyValues(copied, extra)
  }

  override def write: MLWriter = new NNModel.NNModelWriter[T](this)
}

object NNModel extends MLReadable[NNModel[_, _]] {

  import scala.language.existentials
  implicit val format: DefaultFormats.type = DefaultFormats

  private[nnframes] class NNModelReader() extends MLReader[NNModel[_, _]] {
    override def load(path: String): NNModel[_, _] = {
      val (meta, model, typeTag, feaTran) = NNModel.getMetaAndModel(path, sc)
      val featureSize = (meta.metadata \ "featureSize").extract[Seq[Int]].toArray
      val nnModel = typeTag match {
        case "TensorDouble" =>
          new NNModel[Any, Double](model.asInstanceOf[Module[Double]],
            feaTran.asInstanceOf[Transformer[Any, Sample[Double]]])
        case "TensorFloat" =>
          new NNModel[Any, Float](model.asInstanceOf[Module[Float]],
            feaTran.asInstanceOf[Transformer[Any, Sample[Float]]])
        case _ =>
          throw new Exception("Only support float and double for now")
      }

      DefaultParamsWriterWrapper.getAndSetParams(nnModel, meta)
      nnModel
    }
  }

  private[nnframes] def getMetaAndModel(path: String, sc: SparkContext) = {
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

    val ois = new ObjectInputStream(
      new FileInputStream(new Path(path, "featureTransformer").toString))
    val featureTransformer = try {
      ois.readObject.asInstanceOf[Transformer[Any, Any]]
    } finally {
      ois.close()
    }

    (meta, model, typeTag, featureTransformer)
  }

  class NNModelWriter[@specialized(Float, Double) T: ClassTag](
    instance: NNModel[_, T])(implicit ev: TensorNumeric[T]) extends MLWriter {
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
      instance: NNModel[_, T],
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
    DefaultParamsWriterWrapper.saveMetadata(instance, path, sc, Option(extra))
    val (modulePath, weightPath) =
      new Path(path, "module").toString -> new Path(path, "weight").toString
    module.saveModule(modulePath, weightPath, isOverWrite)

    val fos = new FileOutputStream(new Path(path, "featureTransformer").toString)
    val oos = new ObjectOutputStream(fos)
    try {
      oos.writeObject(instance.sampleTransformer)
    } finally {
      oos.close()
    }
  }

  override def read: MLReader[NNModel[_, _]] = new NNModelReader

  override def load(path: String): NNModel[_, _] = read.load(path)
}

