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
package org.apache.spark.ml

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dataset._
import com.intel.analytics.bigdl.models.utils.ModelBroadcast
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import com.intel.analytics.bigdl.utils.T
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasPredictionCol}
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators, _}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Row}

import scala.reflect.ClassTag

private[ml] trait HasBatchSize extends Params {

  final val batchSize: Param[Int] = new Param[Int](this, "batchSize", "batchSize")

  def getBatchSize: Int = $(batchSize)
}

/**
 * Common trait for DLEstimator and DLModel
 */
private[ml] trait DLParams[@specialized(Float, Double) T] extends HasFeaturesCol
  with HasPredictionCol with VectorCompatibility with HasBatchSize {

  /**
   * optimization method to be used. BigDL supports many optimization methods like Adam,
   * SGD and LBFGS. Refer to package com.intel.analytics.bigdl.optim for all the options.
   * Default: SGD
   */
  final val optimMethod = new Param[OptimMethod[T]](this, "optimMethod", "optimMethod")

  def getOptimMethod: OptimMethod[T] = $(optimMethod)

  /**
   * number of max Epoch for the training, an epoch refers to a traverse over the training data
   * Default: 100
   */
  final val maxEpoch = new IntParam(this, "maxEpoch", "number of max Epoch", ParamValidators.gt(0))

  def getMaxEpoch: Int = $(maxEpoch)

  /**
   * learning rate for the optimizer in the DLEstimator.
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

  setDefault(batchSize -> 1)

  /**
   * Validate if feature and label columns are of supported data types.
   * Default: 0
   */
  protected def validateDataType(schema: StructType, colName: String): Unit = {
    val dataTypes = Seq(
      new ArrayType(DoubleType, false),
      new ArrayType(DoubleType, true),
      new ArrayType(FloatType, false),
      new ArrayType(FloatType, true),
      DoubleType,
      FloatType
    ) ++ validVectorTypes

    // TODO use SchemaUtils.checkColumnTypes after convert to 2.0
    val actualDataType = schema(colName).dataType
    require(dataTypes.exists(actualDataType.equals),
      s"Column $colName must be of type equal to one of the following types: " +
        s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.")
  }

  /**
   * Get conversion function to extract data from original DataFrame
   * Default: 0
   */
  protected def getConvertFunc(colType: DataType): (Row, Int) => Seq[AnyVal] = {
    colType match {
      case ArrayType(DoubleType, false) =>
        (row: Row, index: Int) => row.getSeq[Double](index)
      case ArrayType(DoubleType, true) =>
        (row: Row, index: Int) => row.getSeq[Double](index)
      case ArrayType(FloatType, false) =>
        (row: Row, index: Int) => row.getSeq[Float](index)
      case ArrayType(FloatType, true) =>
        (row: Row, index: Int) => row.getSeq[Float](index)
      case DoubleType =>
        (row: Row, index: Int) => Seq[Double](row.getDouble(index))
      case FloatType =>
        (row: Row, index: Int) => Seq[Float](row.getFloat(index))
      case _ =>
        if (colType.typeName.contains("vector")) {
          (row: Row, index: Int) => getVectorSeq(row, colType, index)
        } else {
          throw new IllegalArgumentException(
            s"$colType is not a supported (Unexpected path).")
        }
    }
  }
}


/**
 * [[DLEstimator]] helps to train a BigDL Model with the Spark ML Estimator/Transfomer pattern,
 * thus Spark users can conveniently fit BigDL into Spark ML pipeline.
 *
 * [[DLEstimator]] supports feature and label data in the format of
 * Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
 * org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double and Float.
 *
 * User should specify the feature data dimensions and label data dimensions via the constructor
 * parameters featureSize and labelSize respectively. Internally the feature and label data are
 * converted to BigDL tensors, to further train a BigDL model efficiently.
 *
 * For details usage, please refer to examples in package
 * com.intel.analytics.bigdl.example.MLPipeline
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
 *                    width * height = 28 * 28, featureSize = Array(28, 28).
 * @param labelSize The size (Tensor dimensions) of the label data.
 */
class DLEstimator[@specialized(Float, Double) T: ClassTag](
    @transient val model: Module[T],
    val criterion : Criterion[T],
    val featureSize : Array[Int],
    val labelSize : Array[Int],
    override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends DLEstimatorBase[DLEstimator[T], DLModel[T]] with DLParams[T] {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def setOptimMethod(value: OptimMethod[T]): this.type = set(optimMethod, value)

  def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)
  setDefault(maxEpoch -> 50)

  def setLearningRate(value: Double): this.type = set(learningRate, value)
  setDefault(learningRate -> 1e-3)

  def setLearningRateDecay(value: Double): this.type = set(learningRateDecay, value)
  setDefault(learningRateDecay -> 0.0)

  override def transformSchema(schema : StructType): StructType = {
    validateDataType(schema, $(featuresCol))
    validateDataType(schema, $(labelCol))
    SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, false))
  }

  protected override def internalFit(dataFrame: DataFrame): DLModel[T] = {
    val featureType = dataFrame.schema($(featuresCol)).dataType
    val featureColIndex = dataFrame.schema.fieldIndex($(featuresCol))
    val labelType = dataFrame.schema($(labelCol)).dataType
    val labelColIndex = dataFrame.schema.fieldIndex($(labelCol))

    val featureFunc = getConvertFunc(featureType)
    val labelFunc = getConvertFunc(labelType)

    val featureAndLabel: RDD[(Seq[AnyVal], Seq[AnyVal])] = dataFrame.rdd.map { row =>
      val features = featureFunc(row, featureColIndex)
      val labels = labelFunc(row, labelColIndex)
      (features, labels)
    }

    val samples = featureAndLabel.map { case (f, l) =>
      // convert feature and label data type to the same type with model
      // TODO: investigate to reduce memory consumption during conversion.
      val feature = f.head match {
        case dd: Double => f.asInstanceOf[Seq[Double]].map(ev.fromType(_))
        case ff: Float => f.asInstanceOf[Seq[Float]].map(ev.fromType(_))
      }
      val label = l.head match {
        case dd: Double => l.asInstanceOf[Seq[Double]].map(ev.fromType(_))
        case ff: Float => l.asInstanceOf[Seq[Float]].map(ev.fromType(_))
      }
      (feature, label)
    }.map { case (feature, label) =>
      Sample(Tensor(feature.toArray, featureSize), Tensor(label.toArray, labelSize))
    }

    if(!isDefined(optimMethod)) {
      set(optimMethod, new SGD[T])
    }
    val state = T("learningRate" -> $(learningRate), "learningRateDecay" -> $(learningRateDecay))
    val optimizer = Optimizer(model, samples, criterion, $(batchSize))
      .setState(state)
      .setOptimMethod($(optimMethod))
      .setEndWhen(Trigger.maxEpoch($(maxEpoch)))
    val optimizedModel = optimizer.optimize()

    wrapBigDLModel(optimizedModel, featureSize)
  }

  /**
   * sub classes can extend the method and return required model for different transform tasks
   */
  protected def wrapBigDLModel(m: Module[T], featureSize: Array[Int]): DLModel[T] = {
    val dlModel = new DLModel[T](m, featureSize)
    copyValues(dlModel.setParent(this))
  }

  override def copy(extra: ParamMap): DLEstimator[T] = {
    copyValues(new DLEstimator(model, criterion, featureSize, labelSize), extra)
  }
}

/**
 * [[DLModel]] helps embed a BigDL model into a Spark Transformer, thus Spark users can
 * conveniently merge BigDL into Spark ML pipeline.
 * [[DLModel]] supports feature data in the format of
 * Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
 * org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double and Float.
 * Internally [[DLModel]] use features column as storage of the feature data, and create
 * Tensors according to the constructor parameter featureSize.
 *
 * [[DLModel]] is compatible with both spark 1.5-plus and 2.0 by extending ML Transformer.
 * @param model trainned BigDL models to use in prediction.
 * @param featureSize The size (Tensor dimensions) of the feature data. (e.g. an image may be with
 * featureSize = 28 * 28).
 */
class DLModel[@specialized(Float, Double) T: ClassTag](
    @transient val model: Module[T],
    var featureSize : Array[Int],
    override val uid: String = "DLModel"
    )(implicit ev: TensorNumeric[T])
  extends DLTransformerBase[DLModel[T]] with DLParams[T] with HasBatchSize {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setFeatureSize(value: Array[Int]): this.type = {
    this.featureSize = value
    this
  }

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  def getFeatureSize: Array[Int] = this.featureSize

  /**
   * Perform a prediction on featureCol, and write result to the predictionCol.
   */
  protected override def internalTransform(dataFrame: DataFrame): DataFrame = {
    val featureType = dataFrame.schema($(featuresCol)).dataType
    val featureColIndex = dataFrame.schema.fieldIndex($(featuresCol))
    val featureFunc = getConvertFunc(featureType)
    val sc = dataFrame.sqlContext.sparkContext
    val modelBroadCast = ModelBroadcast[T]().broadcast(sc, model)
    val localBatchSize = $(batchSize)

    val resultRDD = dataFrame.rdd.mapPartitions { rowIter =>
      val localModel = modelBroadCast.value()
      rowIter.grouped(localBatchSize).flatMap { rowBatch =>
        val samples = rowBatch.map { row =>
          val features = featureFunc(row, featureColIndex)
          val featureBuffer = features.head match {
            case dd: Double => features.asInstanceOf[Seq[Double]].map(ev.fromType(_))
            case ff: Float => features.asInstanceOf[Seq[Float]].map(ev.fromType(_))
          }
          Sample(Tensor(featureBuffer.toArray, featureSize))
        }.toIterator
        val predictions = SampleToMiniBatch(localBatchSize).apply(samples).flatMap { batch =>
          val batchResult = localModel.forward(batch.getInput())
          batchResult.toTensor.split(1).map(outputToPrediction)
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
    output.clone().storage().array().map(ev.toType[Double])
  }

  override def transformSchema(schema : StructType): StructType = {
    validateDataType(schema, $(featuresCol))
    SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, false))
  }

  override def copy(extra: ParamMap): DLModel[T] = {
    val copied = new DLModel(model, featureSize, uid).setParent(parent)
    copyValues(copied, extra)
  }
}

// TODO, add save/load
object DLModel {


}

