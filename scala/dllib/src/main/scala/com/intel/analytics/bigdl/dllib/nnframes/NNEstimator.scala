/*
 * Copyright 2018 The Analytics Zoo Authors.
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

import com.intel.analytics.bigdl.{Criterion, Module}
import com.intel.analytics.bigdl.dlframes.{DLEstimator, DLModel}
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import org.apache.spark.sql.types._
import org.apache.spark.sql.Row

import scala.reflect.ClassTag

/**
 * [[NNEstimator]] extends from BigDL DLEstimator, it helps to train a BigDL Model with the Spark
 * ML Estimator/Transfomer pattern, thus Spark users can conveniently fit BigDL into Spark
 * ML pipeline. In Addition, it supports image schema
 *
 * [[NNEstimator]] supports feature and label data in the format of
 * Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
 * org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double and Float.
 *
 * User should specify the feature data dimensions and label data dimensions via the constructor
 * parameters featureSize and labelSize respectively. Internally the feature and label data are
 * converted to BigDL tensors, to further train a BigDL model efficiently.
 *
 * For details usage, please refer to examples in package
 * com.intel.analytics.zoo.pipeline.example.nnframes
 *
 * @param model BigDL module to be optimized
 * @param criterion  BigDL criterion method
 * @param featureSize The size (Tensor dimensions) of the feature data. e.g. an image may be with
 *                    width * height = 28 * 28, featureSize = Array(28, 28).
 * @param labelSize The size (Tensor dimensions) of the label data.
 */
class NNEstimator[T: ClassTag](
    @transient override val model: Module[T],
    override val criterion : Criterion[T],
    override val featureSize : Array[Int],
    override val labelSize : Array[Int],
    override val uid: String = "DLEstimator")(implicit ev: TensorNumeric[T])
  extends DLEstimator[T](model, criterion, featureSize, labelSize, uid) {

  override def validateDataType(schema: StructType, colName: String): Unit = {
    val dataTypes = Seq(
      new ArrayType(DoubleType, false),
      new ArrayType(DoubleType, true),
      new ArrayType(FloatType, false),
      new ArrayType(FloatType, true),
      DoubleType,
      FloatType,
      NNImageSchema.floatSchema
    ) ++ validVectorTypes

    // TODO use SchemaUtils.checkColumnTypes after convert to 2.0
    val actualDataType = schema(colName).dataType
    require(dataTypes.exists(actualDataType.equals),
      s"Column $colName must be of type equal to one of the following types: " +
        s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.")
  }

  override def getConvertFunc(colType: DataType): (Row, Int) => Seq[AnyVal] = {
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
      case NNImageSchema.floatSchema =>
        (row: Row, index: Int) => row.getAs[Row](index).getSeq[Float](5)
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
 * [[NNModel]] helps embed a BigDL model into a Spark Transformer, thus Spark users can
 * conveniently merge BigDL into Spark ML pipeline.
 * [[NNModel]] supports feature data in the format of
 * Array[Double], Array[Float], org.apache.spark.mllib.linalg.{Vector, VectorUDT},
 * org.apache.spark.ml.linalg.{Vector, VectorUDT}, Double and Float.
 * Internally [[NNModel]] use features column as storage of the feature data, and create
 * Tensors according to the constructor parameter featureSize.
 *
 * [[NNModel]] is compatible with both spark 1.5-plus and 2.0 by extending ML Transformer.
 * @param model trainned BigDL models to use in prediction.
 * @param featureSize The size (Tensor dimensions) of the feature data. (e.g. an image may be with
 * featureSize = 28 * 28).
 */
class NNModel[T: ClassTag](
    @transient override val model: Module[T],
    featureSize : Array[Int],
    override val uid: String = "DLModel")(implicit ev: TensorNumeric[T])
  extends DLModel[T](model, featureSize, uid) {

  override def validateDataType(schema: StructType, colName: String): Unit = {
    val dataTypes = Seq(
      new ArrayType(DoubleType, false),
      new ArrayType(DoubleType, true),
      new ArrayType(FloatType, false),
      new ArrayType(FloatType, true),
      DoubleType,
      FloatType,
      NNImageSchema.floatSchema
    ) ++ validVectorTypes

    // TODO use SchemaUtils.checkColumnTypes after convert to 2.0
    val actualDataType = schema(colName).dataType
    require(dataTypes.exists(actualDataType.equals),
      s"Column $colName must be of type equal to one of the following types: " +
        s"${dataTypes.mkString("[", ", ", "]")} but was actually of type $actualDataType.")
  }

  override def getConvertFunc(colType: DataType): (Row, Int) => Seq[AnyVal] = {
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
      case NNImageSchema.floatSchema =>
        (row: Row, index: Int) => row.getAs[Row](index).getSeq[Float](5)
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

