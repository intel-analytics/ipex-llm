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

import org.apache.spark.ml.linalg.{Vector, VectorUDT}
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasFeaturesCol, HasLabelCol, HasPredictionCol}
import org.apache.spark.ml.util.SchemaUtils
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{ArrayType, DoubleType, FloatType, StructType}

private[ml] trait DLParams extends HasFeaturesCol with HasPredictionCol {

  /**
   * validate feature columns data format
   */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    val dataTypes = Seq(
      new ArrayType(DoubleType, false),
      new ArrayType(FloatType, false),
      new VectorUDT)

    // TODO use SchemaUtils.checkColumnTypes after convert to 2.0
    val actualDataType = schema($(featuresCol)).dataType
    require(dataTypes.exists(actualDataType.equals),
      s"Column ${$(featuresCol)} must be of type equal to one of the following types: " +
        s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.")

    SchemaUtils.appendColumn(schema, $(predictionCol), new ArrayType(DoubleType, false))
  }

  /**
   * convert feature columns to array columns if necessary
   */
  protected def toArrayType(dataset: DataFrame): DataFrame = {
    val toArray = udf { (vector: Vector) => vector.toArray }
    var converted = dataset
    if (converted.schema($(featuresCol)).dataType.sameType(new VectorUDT)) {
      val newFeatureCol = getFeatureArrayCol
      converted = converted.withColumn(newFeatureCol, toArray(col($(featuresCol))))
    }

    converted
  }

  protected def getFeatureArrayCol: String = $(featuresCol) + "_Array"

}


/**
 * A wrapper from org.apache.spark.ml.Estimator
 * Extends MLEstimator and override process to gain compatibility with
 * both spark 1.5 and spark 2.0.
 */
private[ml] abstract class DLEstimatorBase
  extends Estimator[DLTransformerBase] with DLParams with HasLabelCol{

  protected def getLabelArrayCol: String = $(labelCol) + "_Array"

  protected def internalFit(dataset: DataFrame): DLTransformerBase

  override def fit(dataset: Dataset[_]): DLTransformerBase = {
    transformSchema(dataset.schema, logging = true)
    internalFit(toArrayType(dataset.toDF()))
  }

  override def transformSchema(schema : StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  /**
   * convert feature and label columns to array columns
   */
  protected override def toArrayType(dataset: DataFrame): DataFrame = {
    var converted = super.toArrayType(dataset)

    // convert label column to array type
    val vec2Array = udf { (vector: Vector) => vector.toArray }
    val num2Array = udf { (d: Double) => Array(d) }
    val labelType = converted.schema($(labelCol)).dataType
    val newLabelCol = getLabelArrayCol

    if (labelType.sameType(new VectorUDT)) {
      converted = converted.withColumn(newLabelCol, vec2Array(col($(labelCol))))
    } else if (labelType.sameType(DoubleType)) {
      converted = converted.withColumn(newLabelCol, num2Array(col($(labelCol))))
    }
    converted
  }

  /**
   * validate both feature and label columns
   */
  protected override def validateAndTransformSchema(schema: StructType): StructType = {
    // validate feature column
    super.validateAndTransformSchema(schema)

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

    SchemaUtils.appendColumn(schema, $(predictionCol), new ArrayType(DoubleType, false))
  }

  override def copy(extra: ParamMap): DLEstimatorBase = defaultCopy(extra)
}
