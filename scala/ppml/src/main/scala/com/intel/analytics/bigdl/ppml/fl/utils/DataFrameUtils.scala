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

package com.intel.analytics.bigdl.ppml.fl.utils


import com.intel.analytics.bigdl
import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import org.apache.logging.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType, MapType, StringType, StructType}
import org.apache.spark.sql.{DataFrame, SparkSession}

import collection.JavaConverters._
import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.fl.FLContext

object DataFrameUtils {
  val logger = LogManager.getLogger(getClass)
  def dataFrameToMiniBatch(df: DataFrame,
                           featureColumn: Array[String] = null,
                           labelColumn: Array[String] = null,
                           hasLabel: Boolean = true,
                           batchSize: Int = 4): bigdl.DataSet[MiniBatch[Float]] = {
    val samples = dataFrameToSampleRDD(df, featureColumn, labelColumn, hasLabel, batchSize)

    DataSet.array(samples.collect()) ->
      SampleToMiniBatch(batchSize, parallelizing = false)
  }
  def dataFrameToSampleRDD(df: DataFrame,
                           featureColumn: Array[String] = null,
                           labelColumn: Array[String] = null,
                           hasLabel: Boolean = true,
                           batchSize: Int = 4): RDD[Sample[Float]] = {
    val spark = FLContext.getSparkSession()
    import spark.implicits._
    var fDf: DataFrame = df
    if (featureColumn != null) {
      val featureList = featureColumn.toList
      fDf = fDf.select(featureList.head, featureList.tail: _*)
    }
    df.columns.foreach(colName => {
      val dataType = getGenericType(df.schema(colName).dataType)
      fDf = dataType match {
        case "scalar" => fDf.withColumn(colName, df.col(colName).cast(FloatType))
        case "complex" => throw new Error("not implemented")
      }

    })
    val samples = fDf.rdd.map(r => {
      var featureNum: Int = 0
      var labelMum: Int = 0
      val inputList = new java.util.ArrayList[Float]()
      val arr = if (featureColumn != null || labelColumn != null) {
        featureColumn.foreach(f => inputList.add(r.getAs[Float](f)))
        if (hasLabel) {
          Log4Error.invalidOperationError(featureColumn != null && labelColumn != null,
            "You must provide both featureColumn and labelColumn " +
              "or neither in training or evaluation.\n" +
              "If neither, the last would be used as label and the rest are the features")
          labelColumn.foreach(f => inputList.add(r.getAs[Float](f)))

        } else {
          Log4Error.invalidOperationError(featureColumn != null,
            "You must provide featureColumn in predict")
        }

        featureNum = featureColumn.length
        labelMum = labelColumn.length
        inputList.asScala.toArray[Float]
      } else {
        logger.warn("featureColumn and labelColumn are not provided, would take the last" +
          "column as label column, and others would be feature columns")
        if (hasLabel) {
          featureNum = r.size - 1
          labelMum = 1
        } else {
          featureNum = r.size
        }

        (0 until r.size).map(i => r.getAs[Float](i)).toArray
      }



      if (hasLabel) {
        Log4Error.invalidOperationError(featureNum + labelMum == r.size,
          "size mismatch")
        val features = Tensor[Float](arr.slice(0, featureNum), Array(featureNum))
        val target = Tensor[Float](arr.slice(featureNum, r.size), Array(labelMum))
        Sample(features, target)
      } else {
        val featureNum = r.size
        val features = Tensor[Float](arr.slice(0, featureNum), Array(featureNum))
        Sample(features)
      }
    })
    samples
  }
  def getGenericType(dataType: DataType): String = {
    dataType match {
      case d =>
        if (d != ArrayType && d != StructType && d != MapType) "scalar"
        else "complex"
    }
  }
  def toTensorArray(df: DataFrame, columns: Array[String] = null): Array[Tensor[Float]] = {
    val col = if (columns == null) df.columns else columns

    var rowNum = 0
    df.collect().map(row => {
      if (rowNum == 0) rowNum = row.length
      val rowArray = new Array[Float](row.length)
      col.indices.foreach(i => {
        rowArray(i) = row.getAs[String](col(i)).toFloat
      })
      Tensor[Float](rowArray, Array(rowArray.length))
    })
  }

  /**
   * Convert a single numerical column to 1D Array
   * @param df
   * @param column
   * @return
   */
  def toArray(df: DataFrame, column: String): Array[Float] = {
    df.collect().map(row => {
      row.getAs[String](column).toFloat
    })
  }

}
