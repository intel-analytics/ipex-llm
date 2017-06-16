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

import scala.reflect.ClassTag
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch}
import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.optim.{Adam, Optimizer, Trigger}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.ml.param.{IntParam, ParamMap, ParamValidators}
import org.apache.spark.sql.DataFrame
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._

/**
 * [[DLEstimator]] helps to train a BigDL Model with the Spark ML Estimator/Transfomer pattern,
 * thus Spark users can conveniently fit BigDL into Spark ML pipeline.
 *
 * The feature column holds the storage (Spark Vectors or array of Floats or Doubles) of
 * the feature data, and user should specify the tensor size(dimensions) via param featureSize.
 * The label column holds the storage (Spark Vectors, array of Floats or Doubles, or Double) of
 * the label data, and user should specify the tensor size(dimensions) via param labelSize.
 * Internally the feature and label data are converted to BigDL tensors, to further train a
 * BigDL model efficiently.
 *
 * For details usage, please refer to example :
 * [[com.intel.analytics.bigdl.example.MLPipeline.DLEstimatorLeNet]]
 *
 * @param model module to be optimized
 * @param criterion  criterion method
 * @param featureSize The size (Tensor dimensions) of the feature data.
 * @param labelSize The size (Tensor dimensions) of the label data.
 */
class DLEstimator[@specialized(Float, Double) T: ClassTag](
    val model: Module[T],
    val criterion : Criterion[T],
    val featureSize : Array[Int],
    val labelSize : Array[Int],
    override val uid: String = "DLEstimator"
  )(implicit ev: TensorNumeric[T]) extends DLEstimatorBase with DLParams with HasBatchSize {

  def setFeaturesCol(featuresColName: String): this.type = set(featuresCol, featuresColName)

  def setLabelCol(labelColName : String) : this.type = set(labelCol, labelColName)

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  def setBatchSize(value: Int): this.type = set(batchSize, value)

  val maxEpoch = new IntParam(this, "maxEpoch", "number of max Epoch", ParamValidators.gt(0))
  setDefault(maxEpoch -> 20)

  def getMaxEpoch: Int = $(maxEpoch)

  def setMaxEpoch(value: Int): this.type = set(maxEpoch, value)

  override def transformSchema(schema : StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  protected override def internalFit(dataFrame: DataFrame): DLTransformerBase = {
    val batches = toMiniBatch(dataFrame)
    val dataset = DataSet.rdd(batches)

    val optimizer = Optimizer(model, dataset, criterion)
      .setOptimMethod(new Adam[T]())
      .setEndWhen(Trigger.maxEpoch($(maxEpoch)))
    val optimizedModel = optimizer.optimize()

    val dlModel = new DLModel[T](optimizedModel, featureSize)
    copyValues(dlModel.setParent(this))
  }

  /**
   * Extract and reassemble data according to batchSize
   */
  private def toMiniBatch(dataFrame: DataFrame) : RDD[MiniBatch[T]] = {
    val featureArrayCol = if (dataFrame.schema($(featuresCol)).dataType.isInstanceOf[ArrayType]) {
      $(featuresCol)
    } else {
      getFeatureArrayCol
    }
    val featureColIndex = dataFrame.schema.fieldIndex(featureArrayCol)

    val labelArrayCol = if (dataFrame.schema($(labelCol)).dataType.isInstanceOf[ArrayType]) {
      $(labelCol)
    } else {
      getLabelArrayCol
    }
    val labelColIndex = dataFrame.schema.fieldIndex(labelArrayCol)

    val featureType = dataFrame.schema(featureArrayCol).dataType.asInstanceOf[ArrayType].elementType
    val labelType = dataFrame.schema(labelArrayCol).dataType.asInstanceOf[ArrayType].elementType

    /**
     * since model data type (float or double) and feature data element type does not necessarily
     * comply, we need to extract data from feature column and convert according to model type.
     */
    val featureAndLabelData = dataFrame.rdd.map { row =>
      val featureData = featureType match {
        case DoubleType =>
          row.getSeq[Double](featureColIndex).toArray.map(ev.fromType(_))
        case FloatType =>
          row.getSeq[Float](featureColIndex).toArray.map(ev.fromType(_))
      }
      require(featureData.length == featureSize.product, s"Data length mismatch:" +
        s" feature data length ${featureData.length}, featureSize: ${featureSize.mkString(", ")}")

      val labelData = labelType match {
        case DoubleType =>
          row.getSeq[Double](labelColIndex).toArray.map(ev.fromType(_))
        case FloatType =>
          row.getSeq[Float](labelColIndex).toArray.map(ev.fromType(_))
      }
      require(featureData.length == featureSize.product, s"Data length mismatch:" +
        s" label data length ${featureData.length}, labelSize: ${featureSize.mkString(", ")}")
      (featureData, labelData)
    }

    featureAndLabelData.mapPartitions { rows =>
      val batches = rows.grouped($(batchSize)).map { batch =>
        val featureData = batch.flatMap(_._1).toArray
        val labelData = batch.flatMap(_._2).toArray
        MiniBatch[T](
          Tensor(featureData, Array(batch.length) ++ featureSize),
          Tensor(labelData, Array(batch.length) ++ labelSize))
      }
      batches
    }
  }

  override def copy(extra: ParamMap): DLEstimator[T] = {
    copyValues(new DLEstimator(model, criterion, featureSize, labelSize), extra)
  }
}
