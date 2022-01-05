/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.utils


import com.intel.analytics.bigdl
import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.FLContext
import org.apache.logging.log4j.LogManager
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{ArrayType, DataType, FloatType, MapType, StringType, StructType}
import org.apache.spark.sql.{Column, DataFrame, SparkSession}
import org.apache.spark.sql.{functions => F}
import collection.JavaConverters._
import collection.JavaConversions._

object DataFrameUtils {
  val logger = LogManager.getLogger(getClass)
  def sampleRDDToMiniBatch(sampleRDD: RDD[Sample[Float]],
                           batchSize: Int = 4): bigdl.DataSet[MiniBatch[Float]] = {
    DataSet.array(sampleRDD.collect()) ->
      SampleToMiniBatch(batchSize, parallelizing = false)
  }
  def dataFrameToArrayRDD(df: DataFrame,
                          featureColumn: Array[String] = null,
                          labelColumn: Array[String] = null,
                          hasLabel: Boolean = true) = {
    // If featureColumn and labelColumn are not specified here, show warning and use default
    val (_featureColumn, _labelColumn) = if (featureColumn == null) {
      logger.warn("featureColumn is not provided, would take the last" +
        "column as label column, and others would be feature columns")
      if (hasLabel) {
        (df.columns.slice(0, df.columns.size - 1), Array(df.columns(df.columns.size - 1)))
      } else {
        (df.columns.slice(0, df.columns.size), Array[String]())
      }
    } else {
      (featureColumn, labelColumn)
    }

    val spark = FLContext.getSparkSession()
    import spark.implicits._
    var fDf: DataFrame = df
    val columnList = _featureColumn.toList ++ _labelColumn.toList

    fDf = fDf.select(columnList.head, columnList.tail: _*)
    // a Map mapping column name to the sorted array of features
    val oneHotMap = new java.util.HashMap[String, Array[String]]()

    // record the DataType of each column
    // If numeric type, cast to Float, if categorical, store to one-hot map
    df.columns.foreach(colName => {
      logger.debug(s"Casting column: $colName")
      if (isNumericColumn(df, colName)) {
        fDf = fDf.withColumn(colName, df.col(colName).cast(FloatType))
      }
      else {
        val featureArray = df.groupBy(colName).count.sort("count")
          .collect().map(r => r.getAs[String](0))
        oneHotMap.put(colName, featureArray)
      }
    })

    // construct to DlLib Tensor for each row, store value in Array first
    val rddOfFeatureAndLabel = fDf.rdd.map(r => {
      val inputList = new java.util.ArrayList[Float]()
      val labelList = new java.util.ArrayList[Float]()
      _featureColumn.foreach(f => {
        if (oneHotMap.contains(f)) {
          // is categorical column, convert to one-hot
          val featureArray = oneHotMap.get(f)
          val oneHotIndex = featureArray.indexOf(r.getAs[String](f))
          val oneHot = new Array[Float](featureArray.length)
          //TODO: we implement this flatten feature for now, will refine in future
          oneHot(oneHotIndex) = 1f
          inputList.addAll(oneHot.toList)
        } else {
          // is numeric column, add value
          inputList.add(r.getAs[Float](f))
        }
      })
      if (hasLabel) _labelColumn.foreach(f => labelList.add(r.getAs[Float](f)))
      (inputList.asScala.toArray[Float], labelList.asScala.toArray[Float])
    })
    rddOfFeatureAndLabel

  }
  def arrayRDDToSampleRDD(arrayRDD: RDD[(Array[Float], Array[Float])]): RDD[Sample[Float]] = {
    val samples = arrayRDD.map {
      case (featureArray, labelArray) =>
        if (labelArray.size != 0) {
          // use Array to construct Tensor
          val features = Tensor[Float](featureArray, Array(featureArray.length))
          val target = Tensor[Float](labelArray, Array(labelArray.length))
          Sample(features, target)
        } else {
          val features = Tensor[Float](featureArray, Array(featureArray.length))
          Sample(features)
        }
    }
    samples
  }
  def arrayRDDToTensorArray(arrayRDD: RDD[(Array[Float], Array[Float])]) = {
    val featureTensorArray = new java.util.ArrayList[Tensor[Float]]()
    val labelTensorArray = new java.util.ArrayList[Float]()
    arrayRDD.collect().map{
      case (featureArray, labelArray) =>
        featureTensorArray.add(Tensor[Float](featureArray, Array(featureArray.size)))
        labelTensorArray.add(labelArray(0))
    }
    (featureTensorArray.asScala.toArray, labelTensorArray.asScala.toArray)
  }
  def getGenericType(dataType: DataType): String = {
    dataType.typeName match {
      case "array"| "struct" | "map" => "complex"
      case "string" => "string"
      case _ => "number"
    }
  }
  def fillNA(df: DataFrame, naValue: String = "NA"): DataFrame = {
    var filledDF = df
    df.columns.foreach{ colName => {
      val dataType = df.schema(colName).dataType
      getGenericType(dataType) match {
        case "complex" =>
          throw new NotImplementedError()
        case "string" =>
          // check if the column is numeric format
          if (isNumericColumn(df, colName)) {
            filledDF = replaceNaWithAverage(filledDF, colName, naValue)
          } else {
            filledDF = replaceNaWithMaxCount(filledDF, colName, naValue)
          }


        case "number" =>
          // if spark infer the DataFrame as number type, then no NA exists
          filledDF
      }
    }}
    filledDF
  }
  def isNumericColumn(df: DataFrame, col: String): Boolean = {
    // Try first 10 elements, if failed, bad luck
    df.select(col).take(10).foreach(row => {
      if (scala.util.Try(row.getAs[String](0).toFloat).isSuccess) return true
    })
    false
  }
  def replaceNaWithAverage(df: DataFrame, col: String, naValue: String): DataFrame = {
    // spark average function will ignore invalid value, thus no need to filter NA here
    val columnAvg = df.select(F.avg(col)).take(1)(0).getAs[Double](0)
    val replaceMap = Map(naValue -> columnAvg.toString)
    df.na.replace(col, replaceMap)
  }
  def replaceNaWithMaxCount(df: DataFrame, col: String, naValue: String): DataFrame = {
    val columnMax =
      df.groupBy(col).count().filter(F.col(col)=!=naValue)
        .orderBy(F.desc("count"))
    if (columnMax.count() != 0) {
      val columnMaxCountValue = columnMax.take(1)(0).getAs[String](0)
      val replaceMap = Map(naValue -> columnMaxCountValue)
      df.na.replace(col, replaceMap)
    } else df
  }
}
