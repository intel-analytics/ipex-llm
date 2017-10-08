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

import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, Dataset, Row}


private[ml] trait DLParams extends HasFeaturesCol with HasPredictionCol {

  /**
   * only validate feature columns here
   */
  protected def validateSchema(schema: StructType): Unit = {
    val dataTypes = Seq(
      new ArrayType(DoubleType, false),
      new ArrayType(FloatType, false),
      new VectorUDT)

    // TODO use SchemaUtils.checkColumnTypes after convert to 2.0
    val actualDataType = schema($(featuresCol)).dataType
    require(dataTypes.exists(actualDataType.equals),
      s"Column ${$(featuresCol)} must be of type equal to one of the following types: " +
        s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.")
  }

  def supportedTypesToSeq(row: Row, colType: DataType, index: Int): Seq[AnyVal] = {
    val featureArr = if (colType == new VectorUDT) {
      row.getAs[Vector](index).toArray.toSeq
    } else if (colType == ArrayType(DoubleType, false)) {
      row.getSeq[Double](index)
    } else if (colType == ArrayType(FloatType, false)) {
      row.getSeq[Float](index)
    } else if (colType == DoubleType) {
      Seq[Double](row.getDouble(index))
    }
    featureArr.asInstanceOf[Seq[AnyVal]]
  }

  protected def getFeatureArrayCol: String = $(featuresCol) + "_Array"
}


/**
 *A wrapper from org.apache.spark.ml.Estimator
 * Extends MLEstimator and override process to gain compatibility with
 * both spark 1.5 and spark 2.0.
 */
private[ml] abstract class DLEstimatorBase[Learner <: DLEstimatorBase[Learner, M],
    M <: DLTransformerBase[M]]
  extends Estimator[M] with DLParams with HasLabelCol {

  protected def getLabelArrayCol: String = $(labelCol) + "_Array"

  protected def internalFit(featureAndLabel: RDD[(Seq[AnyVal], Seq[AnyVal])]): M

  override def fit(dataset: Dataset[_]): M = {
    transformSchema(dataset.schema, logging = true)
    internalFit(toArrayType(dataset.toDF()))
  }

  /**
   * convert feature and label columns to array data
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
   * validate both feature and label columns
   */
  protected override def validateSchema(schema: StructType): Unit = {
    // validate feature column
    super.validateSchema(schema)

    // validate label column
    val dataTypes = Seq(
      new ArrayType(DoubleType, false),
      new ArrayType(FloatType, false),
      new VectorUDT,
      DoubleType)

    // TODO use SchemaUtils.checkColumnTypes after convert to 2.0
    val actualDataType = schema($(labelCol)).dataType
    require(dataTypes.exists(actualDataType.equals),
      s"Column ${$(labelCol)} must be of type equal to one of the following types: " +
        s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.")
  }

  override def copy(extra: ParamMap): Learner = defaultCopy(extra)
}



