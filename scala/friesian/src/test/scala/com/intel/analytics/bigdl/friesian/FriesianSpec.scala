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

package com.intel.analytics.bigdl.friesian

import java.net.URL

import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.friesian.python.PythonFriesian
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, _}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.JavaConverters._
import scala.collection.mutable

class FriesianSpec extends ZooSpecHelper {
  var sqlContext: SQLContext = _
  var sc: SparkContext = _
  val resource: URL = getClass.getClassLoader.getResource("")
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

  "AssignStringIdx limit order by freq" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val stringIdxList = friesian.generateStringIdx(df, cols.toList.asJava, orderByFrequency = true)
    val col4Idx = stringIdxList.get(0).collect().sortBy(_.getInt(1))
    val col5Idx = stringIdxList.get(1).collect().sortBy(_.getInt(1))
    assert(col4Idx(0).getString(0) == "abc")
    assert(col5Idx(0).getString(0) == "aa")
  }

  "mask Int" should "work properly" in {
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

  "mask Long" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", Seq(1L, 2L, 3L, 4L, 5L)),
      Row("alice", Seq(4L, 5L, 6L, 7L, 8L)),
      Row("rose", Seq(1L, 2L))))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(LongType), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dfmasked = friesian.mask(df, Array("history").toList.asJava, 4)
    assert(dfmasked.columns.contains("history_mask"))
    assert(dfmasked.filter("size(history_mask) = 4").count() == 3)
    assert(dfmasked.filter("size(history_mask) = 2").count() == 0)
  }

  "mask Double" should "work properly" in {
    val data: RDD[Row] = sc.parallelize(Seq(
      Row("jack", Seq(1.0, 2.0, 3.0, 4.0, 5.0)),
      Row("alice", Seq(4.0, 5.0, 6.0, 7.0, 8.0)),
      Row("rose", Seq(1.0, 2.0))))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(DoubleType), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    df.show(100, false)
    df.printSchema()
    val dfmasked = friesian.mask(df, Array("history").toList.asJava, 4)
    dfmasked.show(100, false)

    assert(dfmasked.columns.contains("history_mask"))
    assert(dfmasked.filter("size(history_mask) = 4").count() == 3)
    assert(dfmasked.filter("size(history_mask) = 2").count() == 0)
  }


  "mask Long and Int" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", Seq(1, 2, 3, 4, 5), Seq(1L, 2L, 3L, 4L, 5L)),
      Row("alice", Seq(4, 5, 6, 7, 8), Seq(4L, 5L, 6L, 7L, 8L)),
      Row("rose", Seq(1, 2), Seq(1L, 2L))))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(IntegerType), true),
      StructField("history1", ArrayType(LongType), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dfmasked = friesian.mask(df, Array("history", "history1").toList.asJava, 4)
    assert(dfmasked.columns.contains("history_mask"))
    assert(dfmasked.columns.contains("history1_mask"))
    assert(dfmasked.filter("size(history_mask) = 4").count() == 3)
    assert(dfmasked.filter("size(history_mask) = 2").count() == 0)
    assert(dfmasked.filter("size(history1_mask) = 4").count() == 3)
    assert(dfmasked.filter("size(history1_mask) = 2").count() == 0)
  }


  "postpad Int" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", Seq(1, 2, 3, 4, 5), Seq(Seq(1, 2, 3), Seq(1, 2, 3))),
      Row("alice", Seq(4, 5, 6, 7, 8), Seq(Seq(1, 2, 3), Seq(1, 2, 3))),
      Row("rose", Seq(1, 2), Seq(Seq(1, 2, 3), Seq(1, 2, 3)))))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(IntegerType), true),
      StructField("history_list", ArrayType(ArrayType(IntegerType)), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dft = friesian.postPad(df, Array("history", "history_list").toList.asJava, 4)
    dft.schema.fields.map(x => x.dataType).foreach(println(_))

    assert(dft.filter("size(history) = 4").count() == 3)
    assert(dft.filter("size(history_list) = 4").count() == 3)
    assert(dft.filter(dft("name") === "rose").select("history").collect()(0)(0).toString()
      == "WrappedArray(1, 2, 0, 0)")
  }

  "postpad Double" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", Seq(1.0, 2.0, 3.0, 4.0, 5.0), Seq(Seq(1.0, 2.0, 3.0), Seq(1.0, 2.0, 3.0))),
      Row("alice", Seq(4.0, 5.0, 6.0, 7.0, 8.0), Seq(Seq(1.0, 2.0, 3.0), Seq(1.0, 2.0, 3.0))),
      Row("rose", Seq(1.0, 2.0), Seq(Seq(1.0, 2.0, 3.0), Seq(1.0, 2.0, 3.0)))))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(DoubleType), true),
      StructField("history_list", ArrayType(ArrayType(DoubleType)), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dft = friesian.postPad(df, Array("history", "history_list").toList.asJava, 4)
    dft.schema.fields.map(x => x.dataType).foreach(println(_))

    assert(dft.filter("size(history) = 4").count() == 3)
    assert(dft.filter("size(history_list) = 4").count() == 3)
    assert(dft.filter(dft("name") === "rose").select("history").collect()(0)(0).toString()
      == "WrappedArray(1.0, 2.0, 0.0, 0.0)")
  }

  "postpad Long" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("jack", Seq(1L, 2L, 3L, 4L, 5L), Seq(Seq(1L, 2L, 3L), Seq(1L, 2L, 3L))),
      Row("alice", Seq(4L, 5L, 6L, 7L, 8L), Seq(Seq(1L, 2L, 3L), Seq(1L, 2L, 3L))),
      Row("rose", Seq(1L, 2L), Seq(Seq(1L, 2L, 3L), Seq(1L, 2L, 3L)))))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(LongType), true),
      StructField("history_list", ArrayType(ArrayType(LongType)), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dft = friesian.postPad(df, Array("history", "history_list").toList.asJava, 4)
//    dft.schema.fields.map(x => x.dataType).foreach(println(_))
    assert(dft.filter("size(history) = 4").count() == 3)
    assert(dft.filter("size(history_list) = 4").count() == 3)
    assert(dft.filter(dft("name") === "rose").select("history").collect()(0)(0).toString()
      == "WrappedArray(1, 2, 0, 0)")
  }

  "addHisSeq int and float" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("rose", "A", 1, 2.0f, "2019-07-01 12:01:19.000"),
      Row("jack", "B", 1, 2.0f, "2019-07-01 12:01:19.000"),
      Row("jack", "A", 2, 2.0f, "2019-08-01 12:01:19.000"),
      Row("jack", "C", 3, 2.0f, "2019-09-01 12:01:19.000"),
      Row("jack", "D", 4, 1.0f, "2019-10-01 12:01:19.000"),
      Row("jack", "A", 5, 1.0f, "2019-11-01 12:01:19.000"),
      Row("jack", "E", 6, 1.0f, "2019-12-01 12:01:19.000"),
      Row("jack", "F", 7, 0.0f, "2019-12-02 12:01:19.000"),
      Row("alice", "G", 4, 0.0f, "2019-09-01 12:01:19.000"),
      Row("alice", "H", 5, 1.0f, "2019-10-01 12:01:19.000"),
      Row("alice", "I", 6, 0.0f, "2019-11-01 12:01:19.000")))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("category", StringType, true),
      StructField("item", IntegerType, true),
      StructField("other", FloatType, true),
      StructField("time", StringType, true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
      .withColumn("ts", col("time").cast("timestamp").cast("long"))

    df.show(false)
    val dft = friesian.addHistSeq(df, Array("item", "other").toList.asJava, "name", "ts", 1, 4)
    assert(dft.count() == 8)
    assert(dft.filter(df("name") === "alice").count() == 2)
    assert(dft.filter(df("name") === "jack").count() == 6)
    assert(dft.columns.contains("item_hist_seq"))
    assert(dft.columns.contains("other_hist_seq"))
  }

  "addHisSeq double long" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("rose", 1L, 2.0, "2019-07-01 12:01:19.000"),
      Row("jack", 1L, 2.0, "2019-07-01 12:01:19.000"),
      Row("jack", 2L, 2.0, "2019-08-01 12:01:19.000"),
      Row("jack", 3L, 2.0, "2019-09-01 12:01:19.000"),
      Row("jack", 4L, 1.0, "2019-10-01 12:01:19.000"),
      Row("jack", 5L, 1.0, "2019-11-01 12:01:19.000"),
      Row("jack", 6L, 1.0, "2019-12-01 12:01:19.000"),
      Row("jack", 7L, 0.0, "2019-12-02 12:01:19.000"),
      Row("alice", 4L, 0.0, "2019-09-01 12:01:19.000"),
      Row("alice", 5L, 1.0, "2019-10-01 12:01:19.000"),
      Row("alice", 6L, 0.0, "2019-11-01 12:01:19.000")))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("item", LongType, true),
      StructField("other", DoubleType, true),
      StructField("time", StringType, true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
      .withColumn("ts", col("time").cast("timestamp").cast("long"))

    val dft = friesian.addHistSeq(df, Array("item", "other").toList.asJava, "name", "ts", 1, 4)
    assert(dft.count() == 8)
    assert(dft.filter(df("name") === "alice").count() == 2)
    assert(dft.filter(df("name") === "jack").count() == 6)
    assert(dft.columns.contains("item_hist_seq"))
    assert(dft.columns.contains("other_hist_seq"))
  }

  "addHisSeq numSeqs" should "work properly" in {
    val data = sc.parallelize(Seq(
      Row("rose", "A", 1, 2.0f, "2019-07-01 12:01:19.000"),
      Row("jack", "B", 1, 2.0f, "2019-07-01 12:01:19.000"),
      Row("jack", "A", 2, 2.0f, "2019-08-01 12:01:19.000"),
      Row("jack", "C", 3, 2.0f, "2019-09-01 12:01:19.000"),
      Row("jack", "D", 4, 1.0f, "2019-10-01 12:01:19.000"),
      Row("jack", "A", 5, 1.0f, "2019-11-01 12:01:19.000"),
      Row("jack", "E", 6, 1.0f, "2019-12-01 12:01:19.000"),
      Row("jack", "F", 7, 0.0f, "2019-12-02 12:01:19.000"),
      Row("alice", "G", 4, 0.0f, "2019-09-01 12:01:19.000"),
      Row("alice", "H", 5, 1.0f, "2019-10-01 12:01:19.000"),
      Row("alice", "I", 6, 0.0f, "2019-11-01 12:01:19.000")))
    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("category", StringType, true),
      StructField("item", IntegerType, true),
      StructField("other", FloatType, true),
      StructField("time", StringType, true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
      .withColumn("ts", col("time").cast("timestamp").cast("long"))

    val dft = friesian.addHistSeq(df, Array("item", "other").toList.asJava, "name",
      "ts", 1, 4, 1)
    assert(dft.count() == 2)
    assert(dft.filter(df("name") === "alice").count() == 1)
    assert(dft.filter(df("name") === "jack").count() == 1)
    assert(dft.columns.contains("item_hist_seq"))
    assert(dft.columns.contains("other_hist_seq"))
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
    val dft = friesian.addNegSamples(df, 10, negNum = 2)
    assert(dft.filter(col("label") === 1).count() == 6)
    assert(dft.filter(col("label") === 0).count() == 12)
  }

  "addNegHisSeq" should "work properly" in {
    val data: RDD[Row] = sc.parallelize(Seq(
      Row("jack", Seq(1, 2, 3, 4, 5)),
      Row("alice", Seq(4, 5, 6, 7, 8)),
      Row("rose", Seq(1, 2))))

    val schema = StructType(Array(
      StructField("name", StringType, true),
      StructField("history", ArrayType(IntegerType), true)
    ))
    val df = sqlContext.createDataFrame(data, schema)
    val dft = friesian.addNegHisSeq(df, 9, "history", 4)
    assert(dft.select("neg_history").collect().length == 3)
    assert(dft.select("neg_history").rdd.map(r =>
      r.getAs[mutable.WrappedArray[mutable.WrappedArray[Int]]](0)).collect()(0).length == 5)
  }

  "addValueFeatures" should "work properly" in {

    val data: RDD[Row] = sc.parallelize(Seq(
      Row("jack", 2, Seq(1, 2, 3, 4, 5), Seq(Seq(1, 2, 3, 4, 5))),
      Row("alice", 3, Seq(4, 5, 6, 7, 8), Seq(Seq(1, 2, 3, 4, 5))),
      Row("rose", 4, Seq(1, 2), Seq(Seq(1, 2, 3, 4, 5)))))

    val schema: StructType = StructType(Array(
      StructField("name", StringType, true),
      StructField("item", IntegerType, true),
      StructField("item_hist_seq", ArrayType(IntegerType), true),
      StructField("nclk_item_hist_seq", ArrayType(ArrayType(IntegerType)), true)))

    val df = sqlContext.createDataFrame(data, schema)

    val mapping = Map(0 -> 0, 1 -> 0, 2 -> 0, 3 -> 0, 4 -> 1, 5 -> 1, 6 -> 1, 8 -> 2, 9 -> 2)
      .map(x => Row(x._1, x._2)).toArray

    val dictSchema: StructType = StructType(Array(
      StructField("key", IntegerType, true),
      StructField("value", IntegerType, true)))
    val dictDF = sqlContext.createDataFrame(sc.parallelize(mapping), dictSchema)

    val df1 = friesian.addValueFeatures(df,
      Array("item", "item_hist_seq", "nclk_item_hist_seq").toList.asJava,
      dictDF, "item", "category")

    assert(df1.select("category").count == 3)
    assert(df1.select("category_hist_seq").count == 3)
    assert(df1.select("nclk_category_hist_seq").count == 3)
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
