/*
 * Copyright 2018 Analytics Zoo Authors.
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

package com.intel.analytics.zoo.friesian

import java.net.URL

import com.intel.analytics.zoo.common.NNContext
import com.intel.analytics.zoo.pipeline.api.keras.ZooSpecHelper
import com.intel.analytics.zoo.friesian.python.PythonFriesian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.sql.functions._


import scala.collection.JavaConverters._
import scala.collection.mutable

class FriesianSpec extends ZooSpecHelper {
  var sqlContext: SQLContext = _
  var sc: SparkContext = _
  val resource: URL = getClass.getClassLoader.getResource("friesian")
  val friesian = PythonFriesian.ofFloat()

  override def doBefore(): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("FriesianTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "Fill NA int" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_1", "col_2")
    val dfFilled = friesian.fillNa(df, 0, cols.toList.asJava)
    assert(dfFilled.filter(dfFilled("col_1").isNull).count == 0)
    assert(dfFilled.filter(dfFilled("col_2").isNull).count == 0)
  }

  "Fill NA string" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val dfFilled = friesian.fillNa(df, "bb", cols.toList.asJava)
    assert(dfFilled.filter(dfFilled("col_4").isNull).count == 0)
    assert(dfFilled.filter(dfFilled("col_5").isNull).count == 0)
  }

  "Fill NA string int" should "throw exception" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_1", "col_4")
    assertThrows[IllegalArgumentException] {
      friesian.fillNa(df, "bb", cols.toList.asJava)
    }
  }

  "Fill NAInt" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val dfFilled = friesian.fillNaInt(df, 3)
    assert(dfFilled.filter(dfFilled("col_2").isNull).count == 0)
    assert(dfFilled.filter(dfFilled("col_3").isNull).count == 0)
    val dfFilled2 = friesian.fillNaInt(df, 4, List("col_3").asJava)
    assert(dfFilled2.filter(dfFilled2("col_2").isNull).count == 1)
    assert(dfFilled2.filter(dfFilled2("col_3").isNull).count == 0)
  }

  "Clip" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_2", "col_3")
    val dfClip = friesian.clip(df, cols.toList.asJava, 1, 2)
    dfClip.show()
  }

  "AssignStringIdx limit null" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val stringIdxList = friesian.generateStringIdx(df, cols.toList.asJava, null)
    assert(stringIdxList.get(0).count == 3)
    assert(stringIdxList.get(1).count == 2)
  }

  "AssignStringIdx limit int" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val stringIdxList = friesian.generateStringIdx(df, cols.toList.asJava, "2")
    assert(stringIdxList.get(0).count == 1)
    assert(stringIdxList.get(1).count == 1)
  }

  "AssignStringIdx limit dict" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val stringIdxList = friesian.generateStringIdx(df, cols.toList.asJava, "col_4:1,col_5:3")
    assert(stringIdxList.get(0).count == 3)
    assert(stringIdxList.get(1).count == 1)
  }

  "mask" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", Seq(1, 2, 3, 4, 5)),
      Row("alice", Seq(4, 5, 6, 7, 8)),
      Row("rose", Seq(1, 2))))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(IntegerType), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dfmasked = friesian.mask(df, Array("history").toList.asJava, 4)
    assert(dfmasked.columns.contains("history_mask"))
    assert(dfmasked.filter("size(history_mask) = 4").count() == 3)
    assert(dfmasked.filter("size(history_mask) = 2").count() == 0)
  }

  "postpad" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", Seq(1, 2, 3, 4, 5), Seq(Seq(1, 2, 3), Seq(1, 2, 3))),
      Row("alice", Seq(4, 5, 6, 7, 8), Seq(Seq(1, 2, 3), Seq(1, 2, 3))),
      Row("rose", Seq(1, 2), Seq(Seq(1, 2, 3), Seq(1, 2, 3)))))
    val schema = StructType( Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(IntegerType), true),
      StructField("history_list", ArrayType(ArrayType(IntegerType)), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dft = friesian.postPad(df, Array("history", "history_list").toList.asJava, 4)
    dft.show(10, false)
    assert(dft.filter("size(history) = 4").count() == 3)
    assert(dft.filter("size(history_list) = 4").count() == 3)
    assert(dft.filter(dft("name") === "rose").select("history").collect()(0)(0).toString()
      == "WrappedArray(1, 2, 0, 0)")
  }

  "addHisSeq" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", 1, "2019-07-01 12:01:19.000"),
      Row("jack", 2, "2019-08-01 12:01:19.000"),
      Row("jack", 3, "2019-09-01 12:01:19.000"),
      Row("jack", 4, "2019-10-01 12:01:19.000"),
      Row("jack", 5, "2019-11-01 12:01:19.000"),
      Row("jack", 6, "2019-12-01 12:01:19.000"),
      Row("jack", 7, "2019-12-02 12:01:19.000"),
      Row("alice", 4, "2019-09-01 12:01:19.000"),
      Row("alice", 5, "2019-10-01 12:01:19.000"),
      Row("alice", 6, "2019-11-01 12:01:19.000")))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("item", IntegerType, true),
      StructField("time", StringType, true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
           .withColumn("ts", col("time").cast("timestamp").cast("long"))
    val dft = friesian.addHistSeq(df, "name", Array("item").toList.asJava, "ts", 1, 4)
    assert(dft.count() == 8)
    assert(dft.filter(df("name") === "alice").count() == 2)
    assert(dft.filter(df("name") === "jack").count() == 6)
    assert(dft.columns.contains("item_hist_seq"))
  }

  "addNegSamples" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", 1, "2019-07-01 12:01:19.000"),
      Row("jack", 2, "2019-08-01 12:01:19.000"),
      Row("jack", 3, "2019-09-01 12:01:19.000"),
      Row("alice", 4, "2019-09-01 12:01:19.000"),
      Row("alice", 5, "2019-10-01 12:01:19.000"),
      Row("alice", 6, "2019-11-01 12:01:19.000")))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("item", IntegerType, true),
      StructField("time", StringType, true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dft = friesian.addNegSamples(df, 10)
    assert(dft.filter(col("label") === 1).count() == 6)
    assert(dft.filter(col("label") === 0).count() == 6)
  }

  "addNegHisSeq" should "work properly" in {
    val data: RDD[Row] = sc.parallelize(Seq(
      Row("jack", Seq(1, 2, 3, 4, 5)),
      Row("alice", Seq(4, 5, 6, 7, 8)),
      Row("rose", Seq(1, 2))))
    val items: RDD[Row] = sc.parallelize((0  to 2).map(i => Row(i, 0))
      ++ (0  to 2).map(i => Row(i + 3, 1)) ++  (0  to 2).map(i => Row(i + 6, 2)))

    val schema = StructType(Array(
    StructField("name", StringType, true),
    StructField("history", ArrayType(IntegerType), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dft = friesian.addNegHisSeq(df, 9, "history", 4)
    assert(dft.select("neg_item_hist_seq").collect().length == 3)
    assert(dft.select("neg_item_hist_seq").rdd.map(r =>
      r.getAs[mutable.WrappedArray[mutable.WrappedArray[Int]]](0)).collect()(0).length == 5)
  }

  "addLength" should "work properly" in {
    val r = scala.util.Random
    val data: RDD[Row] = sc.parallelize(Seq(
      Row("jack", Seq(1, 2, 3, 4, 5)),
      Row("alice", Seq(4, 5, 6, 7, 8)),
      Row("rose", Seq(1, 2))))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(IntegerType), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dft = friesian.addLength(df, "history")

    assert(dft.columns.contains("history_length"))
    assert(dft.filter("history_length = 2").count() == 1)
    assert(dft.filter("history_length = 5").count() == 2)
  }

  "Fill median" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_1", "col_2")
    val dfFilled = friesian.fillMedian(df, cols.toList.asJava)
    assert(dfFilled.filter(dfFilled("col_1").isNull).count == 0)
    assert(dfFilled.filter(dfFilled("col_2").isNull).count == 0)
  }

  "Median" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_1", "col_2")
    val dfFilled = friesian.median(df, cols.toList.asJava)
    assert(dfFilled.count == 2)
    assert(dfFilled.filter("column == 'col_1'")
      .filter("median == 1.0").count == 1)
  }
}
