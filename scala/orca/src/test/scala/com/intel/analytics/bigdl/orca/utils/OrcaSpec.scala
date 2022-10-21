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

package com.intel.analytics.bigdl.orca

import java.net.URL
import com.intel.analytics.bigdl.dllib.NNContext
import com.intel.analytics.bigdl.orca.python.PythonOrca
import org.apache.logging.log4j.Level
import org.apache.logging.log4j.core.config.Configurator
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{StructField, _}
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext, SparkException}

import scala.collection.JavaConverters._
import scala.collection.mutable
import com.intel.analytics.bigdl.orca.utils._

class OrcaSpec extends ZooSpecHelper {
  var sqlContext: SQLContext = _
  var sc: SparkContext = _
  val resource: URL = getClass.getClassLoader.getResource("")
  val orca = PythonOrca.ofFloat()

  override def doBefore(): Unit = {
    Configurator.setLevel("org", Level.ERROR)
    val conf = new SparkConf().setMaster("local[1]").setAppName("OrcaTest")
    sc = NNContext.initNNContext(conf)
    sqlContext = SQLContext.getOrCreate(sc)
  }

  override def doAfter(): Unit = {
    if (sc != null) {
      sc.stop()
    }
  }

  "AssignStringIdx limit null" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val stringIdxList = orca.generateStringIdx(df, cols.toList.asJava, null)
    TestUtils.conditionFailTest(stringIdxList.get(0).count == 3)
    TestUtils.conditionFailTest(stringIdxList.get(1).count == 2)
  }

  "AssignStringIdx limit int" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val stringIdxList = orca.generateStringIdx(df, cols.toList.asJava, "2")
    TestUtils.conditionFailTest(stringIdxList.get(0).count == 1)
    TestUtils.conditionFailTest(stringIdxList.get(1).count == 1)
  }

  "AssignStringIdx limit dict" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val stringIdxList = orca.generateStringIdx(df, cols.toList.asJava, "col_4:1,col_5:3")
    TestUtils.conditionFailTest(stringIdxList.get(0).count == 3)
    TestUtils.conditionFailTest(stringIdxList.get(1).count == 1)
  }

  "AssignStringIdx limit order by freq" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val stringIdxList = orca.generateStringIdx(df, cols.toList.asJava, orderByFrequency = true)
    val col4Idx = stringIdxList.get(0).collect().sortBy(_.getInt(1))
    val col5Idx = stringIdxList.get(1).collect().sortBy(_.getInt(1))
    TestUtils.conditionFailTest(col4Idx(0).getString(0) == "abc")
    TestUtils.conditionFailTest(col5Idx(0).getString(0) == "aa")
  }

  "Fill NA int" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_1", "col_2")
    val dfFilled = orca.fillNa(df, 0, cols.toList.asJava)
    TestUtils.conditionFailTest(dfFilled.filter(dfFilled("col_1").isNull).count == 0)
    TestUtils.conditionFailTest(dfFilled.filter(dfFilled("col_2").isNull).count == 0)
  }

  "Fill NA string" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_4", "col_5")
    val dfFilled = orca.fillNa(df, "bb", cols.toList.asJava)
    TestUtils.conditionFailTest(dfFilled.filter(dfFilled("col_4").isNull).count == 0)
    TestUtils.conditionFailTest(dfFilled.filter(dfFilled("col_5").isNull).count == 0)
  }

  "Fill NA string int" should "throw exception" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val cols = Array("col_1", "col_4")
    assertThrows[IllegalArgumentException] {
      orca.fillNa(df, "bb", cols.toList.asJava)
    }
  }

  "Fill NAInt" should "work properly" in {
    val path = resource.getFile + "/data1.parquet"
    val df = sqlContext.read.parquet(path)
    val dfFilled = orca.fillNaInt(df, 3)
    TestUtils.conditionFailTest(dfFilled.filter(dfFilled("col_2").isNull).count == 0)
    TestUtils.conditionFailTest(dfFilled.filter(dfFilled("col_3").isNull).count == 0)
    val dfFilled2 = orca.fillNaInt(df, 4, List("col_3").asJava)
    TestUtils.conditionFailTest(dfFilled2.filter(dfFilled2("col_2").isNull).count == 1)
    TestUtils.conditionFailTest(dfFilled2.filter(dfFilled2("col_3").isNull).count == 0)
  }
}
