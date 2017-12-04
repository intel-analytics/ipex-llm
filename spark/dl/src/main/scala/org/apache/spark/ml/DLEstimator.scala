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


trait HasBatchSize extends Params {

  final val batchSize: Param[Int] = new Param[Int](this, "batchSize", "batchSize")

  final def getBatchSize: Int = $(batchSize)
}

/**
 * Common trait for DLEstimator and DLModel
 */
private[ml] trait DLParams[@specialized(Float, Double) T] extends HasFeaturesCol
  with HasPredictionCol with VectorCompatibility {

  /**
   * optimization method to be used. BigDL supports many optimization methods like Adam,
   * SGD and LBFGS. Refer to package com.intel.analytics.bigdl.optim for all the options.
   * Default: SGD
   */
  val optimMethod = new Param[OptimMethod[T]](this, "optimMethod", "optimMethod")

  def getOptimMethod: OptimMethod[T] = $(optimMethod)

  /**
   * number of max Epoch for the training, an epoch refers to a traverse over the training data
   * Default: 100
   */
  val maxEpoch = new IntParam(this, "maxEpoch", "number of max Epoch", ParamValidators.gt(0))

  def getMaxEpoch: Int = $(maxEpoch)

  /**
   * learning rate for the optimizer in the DLEstimator.
   * Default: 0.001
   */
  val learningRate = new DoubleParam(this, "learningRate", "learningRate", ParamValidators.gt(0))

  def getLearningRate: Double = $(learningRate)

  /**
   * learning rate decay for each iteration.
   * Default: 0
   */
  val learningRateDecay = new DoubleParam(this, "learningRateDecay", "learningRateDecay")

  def getLearningRateDecay: Double = $(learningRateDecay)

  protected def validateSchema(schema: StructType, colName: String): Unit = {
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

  protected def supportedTypesToSeq(row: Row, colType: DataType, index: Int): Seq[AnyVal] = {
    val featureArr = if (colType == ArrayType(DoubleType, containsNull = false)) {
      row.getSeq[Double](index)
    } else if (colType == ArrayType(DoubleType, containsNull = true)) {
      row.getSeq[Double](index)
    } else if (colType == ArrayType(FloatType, containsNull = false)) {
      row.getSeq[Float](index)
    } else if (colType == ArrayType(FloatType, containsNull = true)) {
      row.getSeq[Float](index)
    } else if (colType == DoubleType) {
      Seq[Double](row.getDouble(index))
    } else if (colType == FloatType) {
      Seq[Double](row.getFloat(index))
    } else if (colType.typeName.contains("vector")) {
      getVectorSeq(row, colType, index)
    } else {
      throw new IllegalArgumentException(
        s"$colType is not a supported (Unexpected path).")
    }
    featureArr.asInstanceOf[Seq[AnyVal]]
  }
}


/**
 * [[DLEstimator]] helps to train a BigDL Model with the Spark ML Estimator/Transfomer pattern,
 * thus Spark users can conveniently fit BigDL into Spark ML pipeline.
 *
 * [[DLEstimator]] supports feature and label data in the format of Array[Double], Array[Float],
 * org.apache.spark.mllib.linalg.{Vector, VectorUDT} for Spark 1.5, 1.6 and
 * org.apache.spark.ml.linalg.{Vector, VectorUDT} for Spark 2.0+. Also label data can be of
 * DoubleType.
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
    val model: Module[T],
    val criterion : Criterion[T],
    val featureSize : Array[Int],
    val labelSize : Array[Int],
    override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends DLEstimatorBase[DLEstimator[T], DLModel[T]] with DLParams[T] with HasBatchSize {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)
  setDefault(batchSize -> 1)

  def setOptimMethod(value: OptimMethod[T]): this.type = set(optimMethod, value)

  def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)
  setDefault(maxEpoch -> 100)

  def setLearningRate(value: Double): this.type = set(learningRate, value)
  setDefault(learningRate -> 1e-3)

  def setLearningRateDecay(value: Double): this.type = set(learningRateDecay, value)
  setDefault(learningRateDecay -> 0.0)

  override def transformSchema(schema : StructType): StructType = {
    super.validateSchema(schema, $(featuresCol))
    super.validateSchema(schema, $(labelCol))
    SchemaUtils.appendColumn(schema, $(predictionCol), ArrayType(DoubleType, false))
  }

  protected override def internalFit(dataFrame: DataFrame): DLModel[T] = {
    val featureAndLabel: RDD[(Seq[AnyVal], Seq[AnyVal])] = toArrayType(dataFrame)
    val batches = toMiniBatch(featureAndLabel)

    if(!isDefined(optimMethod)) {
      set(optimMethod, new SGD[T])
    }
    val state = T("learningRate" -> $(learningRate), "learningRateDecay" -> $(learningRateDecay))
    val optimizer = Optimizer(model, batches, criterion)
      .setState(state)
      .setOptimMethod($(optimMethod).asInstanceOf[OptimMethod[T]])
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

  /**
   * convert feature and label columns to unified data type.
   */
  protected def toArrayType(dataset: DataFrame): RDD[(Seq[AnyVal], Seq[AnyVal])] = {
    val featureType = dataset.schema($(featuresCol)).dataType
    val featureColIndex = dataset.schema.fieldIndex($(featuresCol))
    val labelType = dataset.schema($(labelCol)).dataType
    val labelColIndex = dataset.schema.fieldIndex($(labelCol))

    dataset.rdd.map { row =>
      val features = supportedTypesToSeq(row, featureType, featureColIndex)
      val labels = supportedTypesToSeq(row, labelType, labelColIndex)
      (features, labels)
    }
  }

  /**
   * Extract and reassemble data according to batchSize
   */
  private def toMiniBatch(
      featureAndLabel: RDD[(Seq[AnyVal], Seq[AnyVal])]): DistributedDataSet[MiniBatch[T]] = {

    val samples = featureAndLabel.map { case (f, l) =>
      // convert feature and label data type to the same type with model
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
    (DataSet.rdd(samples) -> SampleToMiniBatch(${batchSize}))
      .asInstanceOf[DistributedDataSet[MiniBatch[T]]]
  }

  override def copy(extra: ParamMap): DLEstimator[T] = {
    copyValues(new DLEstimator(model, criterion, featureSize, labelSize), extra)
  }
}


/**
 * [[DLModel]] helps embed a BigDL model into a Spark Transformer, thus Spark users can
 * conveniently merge BigDL into Spark ML pipeline.
 * [[DLModel]] supports feature data in the format of Array[Double], Array[Float],
 * org.apache.spark.mllib.linalg.{Vector, VectorUDT} for Spark 1.5, 1.6 and
 * org.apache.spark.ml.linalg.{Vector, VectorUDT} for Spark 2.0+.
 * Internally [[DLModel]] use features column as storage of the feature data, and create
 * Tensors according to the constructor parameter featureSize.
 *
 * [[DLModel]] is compatible with both spark 1.5-plus and 2.0 by extending ML Transformer.
 * @param model trainned BigDL models to use in prediction.
 * @param featureSize The size (Tensor dimensions) of the feature data. (e.g. an image may be with
 * featureSize = 28 * 28).
 */
class DLModel[@specialized(Float, Double) T: ClassTag](
    val model: Module[T],
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
   * @param dataFrame featureData in the format of Seq
   * @return output DataFrame
   */
  protected override def internalTransform(dataFrame: DataFrame): DataFrame = {
    val featureData: RDD[Seq[AnyVal]] = toArrayType(dataFrame)

    model.evaluate()
    val modelBroadCast = ModelBroadcast[T]().broadcast(featureData.sparkContext, model)
    val predictRdd = featureData.map { f =>
      // convert feature data type to the same type with model
      f.head match {
        case dd: Double => f.asInstanceOf[Seq[Double]].map(ev.fromType(_))
        case ff: Float => f.asInstanceOf[Seq[Float]].map(ev.fromType(_))
      }
    }.mapPartitions { feature =>
      val localModel = modelBroadCast.value()
      val tensorBuffer = Tensor[T](Array($(batchSize)) ++ featureSize)
      val batches = feature.grouped($(batchSize))
      batches.flatMap { batch =>
        var i = 1
        // Notice: if the last batch is smaller than the batchSize, we still continue
        // to use this tensorBuffer, but only add the meaningful parts to the result Array.
        batch.foreach { row =>
          tensorBuffer.select(1, i).copy(Tensor(Storage(row.toArray)))
          i += 1
        }
        val output = localModel.forward(tensorBuffer).toTensor[T]
        val predict = batchOutputToPrediction(output)
        predict.take(batch.length)
      }
    }

    val resultRDD = dataFrame.rdd.zip(predictRdd).map { case (row, predict) =>
      Row.fromSeq(row.toSeq ++ Seq(predict))
    }
    val resultSchema = transformSchema(dataFrame.schema)
    dataFrame.sqlContext.createDataFrame(resultRDD, resultSchema)
  }

  protected def batchOutputToPrediction(output: Tensor[T]): Iterable[_] = {
    val predict = output.split(1)
    predict.map(p => p.clone().storage().toArray.map(ev.toType[Double]))
  }

  /**
   * convert feature columns to Seq format
   */
  protected def toArrayType(dataset: DataFrame): RDD[Seq[AnyVal]] = {
    val featureType = dataset.schema($(featuresCol)).dataType
    val featureColIndex = dataset.schema.fieldIndex($(featuresCol))
    dataset.rdd.map { row =>
      val features = supportedTypesToSeq(row, featureType, featureColIndex)
      features
    }
  }

  override def transformSchema(schema : StructType): StructType = {
    validateSchema(schema, $(featuresCol))
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

