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

package com.intel.analytics.bigdl.ppml

import com.intel.analytics.bigdl.dllib.feature.dataset.{LocalDataSet, Sample}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.example.DebugLogger
import com.intel.analytics.bigdl.ppml.utils.DataFrameUtils
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class DataFrameSpec extends FlatSpec with Matchers with BeforeAndAfter with DebugLogger{
  "json DataFrame to DataSet" should "work" in {
    val spark = FLContext.getSparkSession()
    import spark.implicits._
    val df = spark.read.json(this.getClass.getClassLoader.getResource("people.json").getPath)
    val arrayRDD = DataFrameUtils.dataFrameToArrayRDD(df)
    val sampleRDD = DataFrameUtils.arrayRDDToSampleRDD(arrayRDD)
    val dataSet = DataFrameUtils.sampleRDDToMiniBatch(sampleRDD)
    require(dataSet.isInstanceOf[LocalDataSet[Any]], "transformation type wrong")
    require(dataSet.size() == 3, "size wrong")
  }
  "csv DataFrame to DataSet" should "work" in {
    val spark = FLContext.getSparkSession()
    import spark.implicits._
    val df = spark.read.option("header", "true")
      .csv(this.getClass.getClassLoader.getResource("diabetes-test.csv").getPath)
    val arrayRDD = DataFrameUtils.dataFrameToArrayRDD(df)
    val sampleRDD = DataFrameUtils.arrayRDDToSampleRDD(arrayRDD)
    val dataSet = DataFrameUtils.sampleRDDToMiniBatch(sampleRDD)
    require(dataSet.isInstanceOf[LocalDataSet[Any]], "transformation type wrong")
    require(dataSet.size() == 10, "size wrong")
  }
  "fillNA by average" should "work" in {
    val spark = FLContext.getSparkSession()
    val seq = Seq("1", "2", "6", "NA")
    val rdd = spark.sparkContext.parallelize(seq)
    import spark.implicits._
    val df = rdd.toDF()
    val filledDF = DataFrameUtils.fillNA(df)
    val filledArray = filledDF.collect().map(r => r.getAs[String](0).toFloat)
    require(filledArray.sameElements(Array(1f, 2, 6, 3)))
  }
  "fillNA by max count" should "work" in {
    val spark = FLContext.getSparkSession()
    val seq = Seq("a", "a", "b", "NA")
    val rdd = spark.sparkContext.parallelize(seq)
    import spark.implicits._
    val df = rdd.toDF()
    val filledDF = DataFrameUtils.fillNA(df)
    val filledArray = filledDF.collect().map(r => r.getAs[String](0))
    require(filledArray.sameElements(Array("a", "a", "b", "a")))
  }
  "dataframe to array RDD" should "work" in {
    val spark = FLContext.getSparkSession()
    val seq = Seq("1", "2", "6", "NA")
    val rdd = spark.sparkContext.parallelize(seq)
    import spark.implicits._
    val df = rdd.toDF()
    val filledDF = DataFrameUtils.fillNA(df)
    val arrayRDD = DataFrameUtils.dataFrameToArrayRDD(filledDF)
  }
}
