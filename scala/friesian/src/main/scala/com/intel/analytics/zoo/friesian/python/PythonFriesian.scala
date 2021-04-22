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

package com.intel.analytics.zoo.friesian.python

import java.util

import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.zoo.common.PythonZoo
import com.intel.analytics.zoo.friesian.feature.Utils
import java.util.{List => JList}

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions.{col, collect_list, explode, row_number, size, spark_partition_id, struct, udf, log => sqllog}
import org.apache.spark.sql.types.{ArrayType, IntegerType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import scala.collection.mutable.WrappedArray
import scala.util.Random

object PythonFriesian {
  def ofFloat(): PythonFriesian[Float] = new PythonFriesian[Float]()

  def ofDouble(): PythonFriesian[Double] = new PythonFriesian[Double]()
}

class PythonFriesian[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  val numericTypes: List[String] = List("long", "double", "integer")

  def fillNa(df: DataFrame, fillVal: Any = 0, columns: JList[String] = null): DataFrame = {
    val cols = if (columns == null) {
      df.columns
    } else {
      columns.asScala.toArray
    }

    val cols_idx = cols.map(col_n => {
      val idx = df.columns.indexOf(col_n)
      if (idx == -1) {
        throw new IllegalArgumentException(s"The column name ${col_n} does not exist")
      }
      idx
    })

    Utils.fillNaIndex(df, fillVal, cols_idx)
  }

  def fillNaInt(df: DataFrame, fillVal: Int = 0, columns: JList[String] = null): DataFrame = {
    val schema = df.schema
    val allColumns = df.columns

    val cols_idx = if (columns == null) {
      schema.zipWithIndex.filter(pair => pair._1.dataType.typeName == "integer")
        .map(pair => pair._2)
    } else {
      val cols = columns.asScala.toList
      cols.map(col_n => {
        val idx = allColumns.indexOf(col_n)
        if (idx == -1) {
          throw new IllegalArgumentException(s"The column name ${col_n} does not exist")
        }
        if (schema(idx).dataType.typeName != "integer") {
          throw new IllegalArgumentException(s"Only columns of IntegerType are supported, but " +
            s"the type of column ${col_n} is ${schema(idx).dataType.typeName}")
        }
        idx
      })
    }

    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for (idx <- cols_idx) {
        if (row.isNullAt(idx)) {
          origin.update(idx, fillVal)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = df.sparkSession
    spark.createDataFrame(dfUpdated, schema)
  }

  def generateStringIdx(df: DataFrame, columns: JList[String], frequencyLimit: String = null)
  : JList[DataFrame] = {
    var default_limit: Option[Int] = None
    val freq_map = scala.collection.mutable.Map[String, Int]()
    if (frequencyLimit != null) {
      val freq_list = frequencyLimit.split(",")
      for (fl <- freq_list) {
        val frequency_pair = fl.split(":")
        if (frequency_pair.length == 1) {
          default_limit = Some(frequency_pair(0).toInt)
        } else if (frequency_pair.length == 2) {
          freq_map += (frequency_pair(0) -> frequency_pair(1).toInt)
        }
      }
    }
    val cols = columns.asScala.toList
    cols.map(col_n => {
      val df_col = df
        .select(col_n)
        .filter(s"${col_n} is not null")
        .groupBy(col_n)
        .count()
      val df_col_filtered = if (freq_map.contains(col_n)) {
        df_col.filter(s"count >= ${freq_map(col_n)}")
      } else if (default_limit.isDefined) {
        df_col.filter(s"count >= ${default_limit.get}")
      } else {
        df_col
      }

      df_col_filtered.cache()
      val count_list: Array[(Int, Int)] = df_col_filtered.rdd.mapPartitions(Utils.getPartitionSize)
        .collect()
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df_col_filtered.rdd.sparkContext.broadcast(base_dict)

      val windowSpec = Window.partitionBy("part_id").orderBy("count")
      val df_with_part_id = df_col_filtered.withColumn("part_id", spark_partition_id())
      val df_row_number = df_with_part_id.withColumn("row_number", row_number.over(windowSpec))
      val get_label = udf((part_id: Int, row_number: Int) => {
        row_number + base_dict_bc.value.getOrElse(part_id, 0)
      })
      df_row_number
        .withColumn("id", get_label(col("part_id"), col("row_number")))
        .drop("part_id", "row_number", "count")
    }).asJava
  }

  def compute(df: DataFrame): Unit = {
    df.rdd.count()
  }

  def log(df: DataFrame, columns: JList[String], clipping: Boolean = true): DataFrame = {
    var resultDF = df
    val zeroThreshold = (value: Int) => {
      if (value < 0) 0 else value
    }

    val zeroThresholdUDF = udf(zeroThreshold)
    for (i <- 0 until columns.size()) {
      val colName = columns.get(i)
      if (clipping) {
        resultDF = resultDF.withColumn(colName, sqllog(zeroThresholdUDF(col(colName)) + 1))
      } else {
        resultDF = resultDF.withColumn(colName, sqllog(col(colName)))
      }
    }
    resultDF
  }

  def clipMin(df: DataFrame, columns: JList[String], min: Int): DataFrame = {
    var resultDF = df
    val clipFunc = (value: Int) => {
      if (value < min) min else value
    }
    val clipFuncUDF = udf(clipFunc)
    for (i <- 0 until columns.size()) {
      val colName = columns.get(i)
      resultDF = resultDF.withColumn(colName, clipFuncUDF(col(colName)))
    }
    resultDF
  }

  def addHistSeq(df: DataFrame,
                 userCol: String,
                 colNamesin: JList[String],
                 sortCol: String,
                 minLength: Int,
                 maxLength: Int): DataFrame = {
    df.sparkSession.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    val colNames: Array[String] = colNamesin.asScala.toArray
    val schema = ArrayType(StructType(colNames.flatMap(c =>
      Seq(StructField(c, IntegerType), StructField(c + "_history", ArrayType(IntegerType))))))

    val genHisUDF = udf(f = (his_collect: Seq[Row]) => {
      val full_rows = his_collect.sortBy(x => x.getAs[Long](sortCol)).toArray
      val n = full_rows.length
      val result: Seq[Row] = (0 to n - 1).map(i => {
        val lowerBound = if (i < maxLength) {
          0
        } else {
          i - maxLength
        }
        Row.fromSeq(colNames.flatMap(colName =>
          Seq(full_rows(i).getAs[Int](colName),
            full_rows.slice(lowerBound, i).map(row => row.getAs[Int](colName)))))
      })
      result
    }, schema)

    val selectColumns = Seq(col(userCol)) ++
      colNames.flatMap(c =>
        Seq(col("history." + c).as(c),
          col("history." + c + "_history").as(c + "_hist_seq")))
    val collectColumns = colNames.map(c => col(c)) ++ Seq(col(sortCol))
    val filterCondition = colNames.map(c =>
      "size(" + c +  s"_hist_seq) >= $minLength").mkString(" and ")

    df.groupBy(userCol)
      .agg(collect_list(struct(collectColumns: _*)).as("his_collect"))
      .filter(size(col("his_collect")) >= 1)
      .withColumn("history", genHisUDF(col("his_collect")))
      .withColumn("history", explode(col("history")))
      .drop("his_collect")
      .select(selectColumns: _*)
      .filter(filterCondition)
  }

  def mask(df: DataFrame, colNames: JList[String], maxLength: Int): DataFrame = {
    val maskUdf = (history: WrappedArray[Int]) => {
      val n = history.length
      if (maxLength > n) {
        (0 to n - 1).map(_ => 1) ++ (0 to (maxLength - n - 1)).map(_ => 0)
      } else {
        (0 to maxLength - 1).map(_ => 1)
      }
    }

    df.createOrReplaceTempView("tmp")
    df.sqlContext.udf.register("mask", maskUdf)
    val selectStatement = colNames.toArray().map(c => s"mask($c) as $c" + "_mask").mkString(",")

    df.sqlContext.sql(s"select *, $selectStatement from tmp")
  }

  def addNegHisSeq(df: DataFrame, itemSize: Int,
                   item_history_col: String,
                   negNum: Int = 5): DataFrame = {
    val sqlContext = df.sqlContext

    val combinedRDD = df.rdd.map(row => {
      val item_history = row.getAs[WrappedArray[Int]](item_history_col)
      val r = new Random()
      val negItemSeq = Array.ofDim[Int](item_history.length, negNum)
      for (i <- 0 until item_history.length) {
        for (j <- 0 until negNum) {
          var negItem = 0
          do {
            negItem = r.nextInt(itemSize)
          } while (negItem == item_history(i) )
          negItemSeq(i)(j) = negItem
        }
      }
      val result = Row.fromSeq(row.toSeq ++ Array[Any](negItemSeq))
      result
    })

    val newSchema = StructType(df.schema.fields ++ Array(
      StructField("neg_item_hist_seq", ArrayType(ArrayType(IntegerType)))))

   sqlContext.createDataFrame(combinedRDD, newSchema)
  }


  def addNegSamples(df: DataFrame,
                    itemSize: Int,
                    itemCol: String = "item",
                    labelCol: String = "label",
                    negNum: Int = 1): DataFrame = {

    val r = new Random()

    val negativeUdf = udf((itemindex: Int) =>
      (1 to negNum).map(x => {
        var neg = 0
        do {
          neg = r.nextInt(itemSize)
        } while (neg == itemindex)
        neg
      }).map(x => (x, 0)) ++ Seq((itemindex, 1)))

    val columns = df.columns.filter(x => x != itemCol).mkString(",")

    val negativedf = df.withColumn("negative", negativeUdf(col(itemCol)))
      .withColumn("negative", explode(col("negative")))

    negativedf.createOrReplaceTempView("tmp")

    df.sqlContext
      .sql(s"select $columns , negative._1 as item, negative._2 as $labelCol from tmp")
  }

  def postPad(df: DataFrame, colNames: JList[String], maxLength: Int = 100): DataFrame = {

    val padArrayUdf = (history: WrappedArray[Int]) => {
      val n = history.length

      if (maxLength > n) {
        history ++ ((0 to maxLength - n - 1).map(_ => 0))
      } else {
        history.slice(n - maxLength, n)
      }
    }

    val padMaxtrixUdf = (history: WrappedArray[WrappedArray[Int]]) => {
      val n = history.length
      val result: Seq[Seq[Int]] = if (maxLength > n) {
        val hishead = history(0)
        val padArray: Seq[Seq[Int]] =
          (0 to maxLength - n - 1).map(_ => (0 to hishead.length - 1).map(_ => 0))
        history ++ padArray
      } else {
        history.slice(n - maxLength, n)
      }
      result
    }

    df.createOrReplaceTempView("tmp")
    df.sqlContext.udf.register("post_pad_array", padArrayUdf)
    df.sqlContext.udf.register("post_pad_matrix", padMaxtrixUdf)

    val selectStatement = df.schema.fields
      .filter(x => colNames.contains(x.name)).map(x => {
      val c = x.name
      if(x.dataType == ArrayType(IntegerType)) {
        s"post_pad_array($c) as $c"
      } else {
        s"post_pad_matrix($c) as $c"
      }
    }).mkString(",")
    val leftCols = df.columns.filter(x => !colNames.contains(x)).mkString(",")

    df.sqlContext.sql(s"select $leftCols, $selectStatement from tmp")
  }

  def addLength(df: DataFrame, colName: String): DataFrame = {
    df.withColumn(colName + "_length", size(col(colName)))
  }

}
