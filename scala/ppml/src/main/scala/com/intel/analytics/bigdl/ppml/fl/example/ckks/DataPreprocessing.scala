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
package com.intel.analytics.bigdl.ppml.fl.example.ckks

import com.intel.analytics.bigdl.DataSet
import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch, TensorSample}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{array, col, udf}
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}

import scala.collection.mutable.ArrayBuffer
import scala.reflect.ClassTag

class DataPreprocessing(spark: SparkSession,
                        trainDataPath: String,
                        testDataPath: String,
                        clientId: Int) extends Serializable {
  case class Record(
                     age: Int,
                     workclass: String,
                     fnlwgt: Int,
                     education: String,
                     education_num: Int,
                     marital_status: String,
                     occupation: String,
                     relationship: String,
                     race: String,
                     gender: String,
                     capital_gain: Int,
                     capital_loss: Int,
                     hours_per_week: Int,
                     native_country: String,
                     income_bracket: String
                   )
  val batchSize = 8192
  val modelType = "wide"

  val recordSchema = StructType(Array(
    StructField("age", IntegerType, false),
    StructField("workclass", StringType, false),
    StructField("fnlwgt", IntegerType, false),
    StructField("education", StringType, false),
    StructField("education_num", IntegerType, false),
    StructField("marital_status", StringType, false),
    StructField("occupation", StringType, false),
    StructField("relationship", StringType, false),
    StructField("race", StringType, false),
    StructField("gender", StringType, false),
    StructField("capital_gain", IntegerType, false),
    StructField("capital_loss", IntegerType, false),
    StructField("hours_per_week", IntegerType, false),
    StructField("native_country", StringType, false),
    StructField("income_bracket", StringType, false)
  ))


  case class RecordSample[T: ClassTag](sample: Sample[T])

  Configurator.setLevel("org", Level.ERROR)

  def loadCensusData():
    (DataSet[MiniBatch[Float]], DataSet[MiniBatch[Float]]) = {
      val training = spark.sparkContext
        .textFile(trainDataPath)
        .map(_.split(",").map(_.trim))
        .filter(_.size == 15).map(array =>
        Row(
          array(0).toInt, array(1), array(2).toInt, array(3), array(4).toInt,
          array(5), array(6), array(7), array(8), array(9),
          array(10).toInt, array(11).toInt, array(12).toInt, array(13), array(14)
        )
      )

      val validation = spark.sparkContext
        .textFile(testDataPath)
        .map(_.dropRight(1))  // remove dot at the end of each line in adult.test
        .map(_.split(",").map(_.trim))
        .filter(_.size == 15).map(array =>
        Row(
          array(0).toInt, array(1), array(2).toInt, array(3), array(4).toInt,
          array(5), array(6), array(7), array(8), array(9),
          array(10).toInt, array(11).toInt, array(12).toInt, array(13), array(14)
        ))

    val (trainDf, valDf) = (spark.createDataFrame(training, recordSchema),
      spark.createDataFrame(validation, recordSchema))

    println(trainDf.show(10))
    val localColumnInfo = if (clientId == 1) {
      ColumnFeatureInfo(
        wideBaseCols = Array("edu", "occ", "age_bucket"),
        wideBaseDims = Array(16, 1000, 11),
        wideCrossCols = Array("edu_occ", "age_edu_occ"),
        wideCrossDims = Array(1000, 1000),
        indicatorCols = Array("work", "edu", "mari"),
        indicatorDims = Array(9, 16, 7),
        embedCols = Array("occ"),
        embedInDims = Array(1000),
        embedOutDims = Array(8),
        continuousCols = Array("age", "education_num"))
    } else {
      ColumnFeatureInfo(
        wideBaseCols = Array("rela", "work", "mari"),
        wideBaseDims = Array(6, 9, 7),
        indicatorCols = Array("rela"),
        indicatorDims = Array(6),
        // TODO: the error may well be the missed field here
        continuousCols = Array("capital_gain",
          "capital_loss", "hours_per_week"))
    }


    val isImplicit = false
    val trainpairFeatureRdds =
      assemblyFeature(isImplicit, trainDf, localColumnInfo, modelType)

    val validationpairFeatureRdds =
      assemblyFeature(isImplicit, valDf, localColumnInfo, modelType)

    val trainDataset = DataSet.array(
      trainpairFeatureRdds.map(_.sample).collect()) -> SampleToMiniBatch[Float](batchSize)
    val validationDataset = DataSet.array(
      validationpairFeatureRdds.map(_.sample).collect()) -> SampleToMiniBatch[Float](batchSize)
    (trainDataset, validationDataset)
  }

  case class ColumnFeatureInfo(wideBaseCols: Array[String] = Array[String](),
                               wideBaseDims: Array[Int] = Array[Int](),
                               wideCrossCols: Array[String] = Array[String](),
                               wideCrossDims: Array[Int] = Array[Int](),
                               indicatorCols: Array[String] = Array[String](),
                               indicatorDims: Array[Int] = Array[Int](),
                               embedCols: Array[String] = Array[String](),
                               embedInDims: Array[Int] = Array[Int](),
                               embedOutDims: Array[Int] = Array[Int](),
                               continuousCols: Array[String] = Array[String](),
                               label: String = "label") extends Serializable {
    override def toString: String = {
      "wideBaseCols:" + wideBaseCols.mkString(",") + "\n" +
        "wideBaseDims:" + wideBaseDims.mkString(",") + "\n" +
        "wideCrossCols:" + wideCrossCols.mkString(",") + "\n" +
        "wideCrossDims:" + wideCrossDims.mkString(",") + "\n" +
        "indicatorCols:" + indicatorCols.mkString(",") + "\n" +
        "indicatorDims:" + indicatorDims.mkString(",") + "\n" +
        "embedCols:" + embedCols.mkString(",") + "\n" +
        "embedInDims:" + embedInDims.mkString(",") + "\n" +
        "embedOutDims:" + embedOutDims.mkString(",") + "\n" +
        "continuousCols:" + continuousCols.mkString(",") + "\n" +
        "label:" + label

    }
  }

  def categoricalFromVocabList(vocabList: Array[String]): (String) => Int = {
    val func = (sth: String) => {
      val default: Int = 0
      val start: Int = 1
      if (vocabList.contains(sth)) vocabList.indexOf(sth) + start
      else default
    }
    func
  }

  def buckBuckets(bucketSize: Int)(col: String*): Int = {
    Math.abs(col.reduce(_ + "_" + _).hashCode()) % bucketSize + 0
  }

  def bucketizedColumn(boundaries: Array[Float]): Float => Int = {
    col1: Float => {
      var index = 0
      while (index < boundaries.length && col1 >= boundaries(index)) {
        index += 1
      }
      index
    }
  }

  def getDeepTensor(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
    val deepColumns1 = columnInfo.indicatorCols
    val deepColumns2 = columnInfo.embedCols ++ columnInfo.continuousCols
    val deepLength = columnInfo.indicatorDims.sum + deepColumns2.length
    val deepTensor = Tensor[Float](deepLength).fill(0)

    // setup indicators
    var acc = 0
    (0 to deepColumns1.length - 1).map {
      i =>
        val index = r.getAs[Int](columnInfo.indicatorCols(i))
        val accIndex = if (i == 0) index
        else {
          acc = acc + columnInfo.indicatorDims(i - 1)
          acc + index
        }
        deepTensor.setValue(accIndex + 1, 1)
    }

    // setup embedding and continuous
    (0 to deepColumns2.length - 1).map {
      i =>
        deepTensor.setValue(i + 1 + columnInfo.indicatorDims.sum,
          r.getAs[Int](deepColumns2(i)).toFloat)
    }
    deepTensor
  }

  def getWideTensor(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
    val wideColumns = columnInfo.wideBaseCols ++ columnInfo.wideCrossCols
    val wideDims = columnInfo.wideBaseDims ++ columnInfo.wideCrossDims
    val wideLength = wideColumns.length
    var acc = 0
    val indices: Array[Int] = (0 to wideLength - 1).map(i => {
      val index = r.getAs[Int](wideColumns(i))
      if (i == 0) {
        index
      }
      else {
        acc = acc + wideDims(i - 1)
        acc + index
      }
    }).toArray
    val values = indices.map(_ => 1.0f)
    val shape = Array(wideDims.sum)

    Tensor.sparse(Array(indices), values, shape)
  }

  def getWideTensorSequential(r: Row, columnInfo: ColumnFeatureInfo): Tensor[Float] = {
    val wideColumns = columnInfo.wideBaseCols ++ columnInfo.wideCrossCols
    val wideDims = columnInfo.wideBaseDims ++ columnInfo.wideCrossDims
    val wideLength = wideColumns.length
    var acc = 0
    val indices: Array[Int] = (0 to wideLength - 1).map(i => {
      val index = r.getAs[Int](wideColumns(i))
      if (i == 0) index
      else {
        acc = acc + wideDims(i - 1)
        acc + index
      }
    }).toArray
    val values = indices.map(_ => 1.0f)
    val shape = Array(wideDims.sum)

    Tensor.sparse(Array(indices), values, shape)
  }

  def row2SampleSequential(r: Row,
                           columnInfo: ColumnFeatureInfo,
                           modelType: String): Sample[Float] = {
    val wideTensor: Tensor[Float] = getWideTensorSequential(r, columnInfo)
    val deepTensor: Tensor[Float] = getDeepTensor(r, columnInfo)

    val label = if (clientId == 2) {
      val l = r.getAs[Int](columnInfo.label)
      val label = Tensor[Float](T(l))
      Array(label.resize(1))
    } else {
      Array[Tensor[Float]]()
    }



    modelType match {
      case "wide_n_deep" =>
        TensorSample[Float](Array(wideTensor, deepTensor), label)
      case "wide" =>
        TensorSample[Float](Array(wideTensor), label)
      case "deep" =>
        TensorSample[Float](Array(deepTensor), label)
      case _ =>
        throw new IllegalArgumentException("unknown type")
    }
  }

  // convert features to RDD[Sample[Float]]
  def assemblyFeature(isImplicit: Boolean = false,
                      dataDf: DataFrame,
                      columnInfo: ColumnFeatureInfo,
                      modelType: String): RDD[RecordSample[Float]] = {
    val educationVocab = Array("Bachelors", "HS-grad", "11th", "Masters", "9th",
      "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
      "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
      "Preschool", "12th") // 16
    val maritalStatusVocab = Array("Married-civ-spouse", "Divorced", "Married-spouse-absent",
      "Never-married", "Separated", "Married-AF-spouse", "Widowed")
    val relationshipVocab = Array("Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
      "Other-relative") // 6
    val workclassVocab = Array("Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
      "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked") // 9
    val genderVocab = Array("Female", "Male")

    val ages = Array(18f, 25, 30, 35, 40, 45, 50, 55, 60, 65)

    val educationVocabUdf = udf(categoricalFromVocabList(educationVocab))
    val maritalStatusVocabUdf = udf(categoricalFromVocabList(maritalStatusVocab))
    val relationshipVocabUdf = udf(categoricalFromVocabList(relationshipVocab))
    val workclassVocabUdf = udf(categoricalFromVocabList(workclassVocab))
    val genderVocabUdf = udf(categoricalFromVocabList(genderVocab))

    val bucket1Udf = udf(buckBuckets(1000)(_: String))
    val bucket2Udf = udf(buckBuckets(1000)(_: String, _: String))
    val bucket3Udf = udf(buckBuckets(1000)(_: String, _: String, _: String))

    val ageBucketUdf = udf(bucketizedColumn(ages))

    val incomeUdf = udf((income: String) => if (income == ">50K" || income == ">50K.") 1 else 0)

    val data = dataDf
      .withColumn("age_bucket", ageBucketUdf(col("age")))
      .withColumn("edu_occ", bucket2Udf(col("education"), col("occupation")))
      .withColumn("age_edu_occ", bucket3Udf(col("age_bucket"), col("education"),
        col("occupation")))
      .withColumn("edu", educationVocabUdf(col("education")))
      .withColumn("mari", maritalStatusVocabUdf(col("marital_status")))
      .withColumn("rela", relationshipVocabUdf(col("relationship")))
      .withColumn("work", workclassVocabUdf(col("workclass")))
      .withColumn("occ", bucket1Udf(col("occupation")))
      .withColumn("label", incomeUdf(col("income_bracket")))
    val rddOfSample = data.rdd.map(r => {
      RecordSample(row2SampleSequential(r, columnInfo, modelType))
    })
    rddOfSample
  }

}
