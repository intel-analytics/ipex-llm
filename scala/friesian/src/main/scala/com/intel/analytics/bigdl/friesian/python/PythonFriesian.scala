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

package com.intel.analytics.bigdl.friesian.python

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.PythonZoo
import java.util.{List => JList}

import com.intel.analytics.bigdl.friesian.feature.Utils
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types.{ArrayType, DoubleType, IntegerType, LongType, StringType, StructField, StructType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{col, rand, row_number, spark_partition_id, udf, log => sqllog}

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import scala.collection.mutable.WrappedArray
import scala.math.pow

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

    val cols_idx = Utils.getIndex(df, cols)

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

  def generateStringIdx(df: DataFrame, columns: JList[String], frequencyLimit: String = null,
                        orderByFrequency: Boolean = false)
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
      val df_col_ordered = if (orderByFrequency) {
        df_col.orderBy(col("count").desc)
      } else df_col
      val df_col_filtered = if (freq_map.contains(col_n)) {
        df_col_ordered.filter(s"count >= ${freq_map(col_n)}")
      } else if (default_limit.isDefined) {
        df_col_ordered.filter(s"count >= ${default_limit.get}")
      } else {
        df_col_ordered
      }

      df_col_filtered.cache()
      val count_list: Array[(Int, Int)] = df_col_filtered.rdd.mapPartitions(Utils.getPartitionSize)
        .collect().sortBy(_._1)  // further guarantee prior partitions are given smaller indices.
      val base_dict = scala.collection.mutable.Map[Int, Int]()
      var running_sum = 0
      for (count_tuple <- count_list) {
        base_dict += (count_tuple._1 -> running_sum)
        running_sum += count_tuple._2
      }
      val base_dict_bc = df_col_filtered.rdd.sparkContext.broadcast(base_dict)

      val windowSpec = Window.partitionBy("part_id").orderBy(col("count").desc)
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
    val colsIdx = Utils.getIndex(df, columns.asScala.toArray)
    for(i <- 0 until columns.size()) {
      val colName = columns.get(i)
      val colType = df.schema(colsIdx(i)).dataType.typeName
      if (!Utils.checkColumnNumeric(df, colName)) {
        throw new IllegalArgumentException(s"Unsupported data type $colType of column $colName")
      }
    }

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

  def clip(df: DataFrame, columns: JList[String], min: Any = null, max: Any = null):
  DataFrame = {
    if (min == null && max == null) {
      throw new IllegalArgumentException(s"min and max cannot be both null")
    }
    var resultDF = df
    val cols = columns.asScala.toArray
    val colsType = Utils.getIndex(df, cols).map(idx => df.schema(idx).dataType.typeName)
    (cols zip colsType).foreach(nameAndType => {
      if (!Utils.checkColumnNumeric(df, nameAndType._1)) {
        throw new IllegalArgumentException(s"Unsupported data type ${nameAndType._2} of " +
          s"column ${nameAndType._1}")
      }
    })

    for(i <- 0 until columns.size()) {
      val colName = columns.get(i)
      val colType = colsType(i)

      val minVal = Utils.castNumeric(min, colType)
      val maxVal = Utils.castNumeric(max, colType)

      val clipFuncUDF = colType match {
        case "long" => udf(Utils.getClipFunc[Long](minVal, maxVal, colType))
        case "integer" => udf(Utils.getClipFunc[Int](minVal, maxVal, colType))
        case "double" => udf(Utils.getClipFunc[Double](minVal, maxVal, colType))
        case _ => throw new IllegalArgumentException(s"Unsupported data type $colType of column" +
          s" $colName")
      }
      resultDF = resultDF.withColumn(colName, clipFuncUDF(col(colName)))
    }
    resultDF
  }

  def crossColumns(df: DataFrame,
                   crossCols: JList[JList[String]],
                   bucketSizes: JList[Int]): DataFrame = {
    def crossColumns(bucketSize: Int) = udf((cols: WrappedArray[Any]) => {
      Utils.hashBucket(cols.mkString("_"), bucketSize = bucketSize)
    })

    var resultDF = df
    for (i <- 0 until crossCols.size()) {
      resultDF = resultDF.withColumn(crossCols.get(i).asScala.toList.mkString("_"),
        crossColumns(bucketSizes.get(i))(
          array(crossCols.get(i).asScala.toArray.map(x => col(x)): _*)
        ))
    }
    resultDF
  }

  def addHistSeq(df: DataFrame,
                 cols: JList[String],
                 userCol: String,
                 timeCol: String,
                 minLength: Int,
                 maxLength: Int,
                 nunSeqs: Int = Int.MaxValue): DataFrame = {

    df.sparkSession.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    val colNames: Array[String] = cols.asScala.toArray

    val colsWithType = df.schema.fields.filter(x => x.name != userCol)
    val schema = ArrayType(StructType(colsWithType.flatMap(c =>
      if (colNames.contains(c.name)) {
        Seq(c, StructField(c.name + "_hist_seq", ArrayType(c.dataType)))
      } else {
        Seq(c)
      })))

    val genHisUDF = udf(f = (his_collect: Seq[Row]) => {

      val full_rows: Array[Row] = his_collect.sortBy(x => x.getAs[Long](timeCol)).toArray
      val n = full_rows.length

      val couples: Seq[(Int, Int)] = {
        (minLength to n - 1).map(i => {
          val lowerBound = if (i < maxLength) {
            0
          } else {
            i - maxLength
          }
          (lowerBound, i)
        })
      }


      val result: Seq[Row] = couples.takeRight(nunSeqs).map(x => {
        val rowValue: Array[Any] = colsWithType.flatMap(col => {
          if (colNames.contains(col.name)) {
            col.dataType.typeName match {
              case "integer" => Utils.get1row[Int](full_rows, col.name, x._2, x._1)
              case "double" => Utils.get1row[Double](full_rows, col.name, x._2, x._1)
              case "float" => Utils.get1row[Float](full_rows, col.name, x._2, x._1)
              case "long" => Utils.get1row[Long](full_rows, col.name, x._2, x._1)
              case _ => throw new IllegalArgumentException(
                s"Unsupported data type ${col.dataType.typeName} " +
                  s"of column ${col.name} in add_hist_seq")
            }
          } else {
            val colValue: Any = full_rows(x._2).getAs(col.name)
            Seq(colValue)
          }
        })
        Row.fromSeq(rowValue)
      })

      result
    }, schema)

    val allColumns = colsWithType.map(x => col(x.name))
    df.groupBy(userCol).agg(collect_list(struct(allColumns: _*)).as("friesian_his_collect"))
      .filter("size(friesian_his_collect) > 1")
      .withColumn("friesian_history", explode(genHisUDF(col("friesian_his_collect"))))
      .select(userCol, "friesian_history.*")
  }

  def addValueFeatures(df: DataFrame, cols: JList[String], dictDF: DataFrame,
                       key: String, value: String): DataFrame = {

    val mapScala = dictDF.rdd.map(r => (r.getInt(0), r.getInt(1))).collect().toMap
    val sc = df.sparkSession.sparkContext
    val mapBr: Broadcast[Map[Int, Int]] = sc.broadcast(mapScala)

    var tmpDF = df
    for(col <- cols.asScala.toList) {
      tmpDF = Utils.addValueSingleCol(tmpDF, col, mapBr, key, value)
    }

    tmpDF
  }


  def mask(df: DataFrame, cols: JList[String], maxLength: Int): DataFrame = {

    var maskDF = df

    val maskUdf = udf(Utils.maskArr)

    cols.asScala.toList.foreach(c => {
      maskDF = maskDF.withColumn(c + "_mask", maskUdf(lit(maxLength), col(c)))
    })

    maskDF
  }


  def addNegHisSeq(df: DataFrame, itemSize: Int,
                   historyCol: String,
                   negNum: Int = 5): DataFrame = {

    df.sparkSession.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    val itemType = df.select(explode(col(historyCol))).schema.fields(0).dataType
    require(itemType.typeName == "integer", throw new IllegalArgumentException(
      s"Unsupported data type ${itemType.typeName} " +
        s"of column ${historyCol} in add_neg_hist_seq"))
    val schema = ArrayType(ArrayType(itemType))

    val negativeUdf = udf(Utils.addNegativeList(negNum, itemSize), schema)

    df.withColumn("neg_" + historyCol, negativeUdf(col(historyCol)))
  }

  def addNegSamples(df: DataFrame,
                    itemSize: Int,
                    itemCol: String = "item",
                    labelCol: String = "label",
                    negNum: Int = 1): DataFrame = {

    df.sparkSession.conf.set("spark.sql.legacy.allowUntypedScalaUDF", "true")
    val itemType = df.select(itemCol).schema.fields(0).dataType
    require(itemType.typeName == "integer", throw new IllegalArgumentException(
      s"Unsupported data type ${itemType.typeName} " +
        s"of column ${itemCol} in add_negative_samples"))
    val schema = ArrayType(StructType(Seq(StructField(itemCol, itemType),
      StructField(labelCol, itemType))))

    val negativeUdf = udf(Utils.addNegtiveItem(negNum, itemSize), schema)

    val negativedf = df.withColumn("item_label", explode(negativeUdf(col(itemCol))))

    val selectColumns = df.columns.filter(x => x != itemCol)
      .map(ele => col(ele)) ++ Seq(col("item_label.*"))

    negativedf.select(selectColumns: _*)
  }

  def postPad(df: DataFrame, cols: JList[String], maxLength: Int = 100): DataFrame = {

    val colFields = df.schema.fields.filter(x => cols.contains(x.name))

    var paddedDF = df

    colFields.foreach(c => {
      val dataType = c.dataType
      val padUdf = dataType match {
        case ArrayType(IntegerType, _) => udf(Utils.padArr[Int])
        case ArrayType(LongType, _) => udf(Utils.padArr[Long])
        case ArrayType(DoubleType, _) => udf(Utils.padArr[Double])
        case ArrayType(ArrayType(IntegerType, _), _) => udf(Utils.padMatrix[Int])
        case ArrayType(ArrayType(LongType, _), _) => udf(Utils.padMatrix[Long])
        case ArrayType(ArrayType(DoubleType, _), _) => udf(Utils.padMatrix[Double])
        case _ => throw new IllegalArgumentException(
          s"Unsupported data type $dataType of column $c in pad")
      }
      paddedDF = paddedDF.withColumn(c.name, padUdf(lit(maxLength), col(c.name)))
    })

    paddedDF
  }

  def fillMedian(df: DataFrame, columns: JList[String] = null): DataFrame = {
    val cols = if (columns == null) {
      df.columns.filter(column => Utils.checkColumnNumeric(df, column))
    } else {
      columns.asScala.toArray
    }

    val colsIdx = Utils.getIndex(df, cols)
    val medians = Utils.getMedian(df, cols)
    val idxMedians = (colsIdx zip medians).map(idxMedian => {
      if (idxMedian._2 == null) {
        throw new IllegalArgumentException(
          s"Cannot compute the median of column ${cols(idxMedian._1)} " +
            s"since it contains only null values.")
      }
      val colType = df.schema(idxMedian._1).dataType.typeName
      colType match {
        case "long" => (idxMedian._1, idxMedian._2.asInstanceOf[Double].longValue)
        case "integer" => (idxMedian._1, idxMedian._2.asInstanceOf[Double].intValue)
        case "double" => (idxMedian._1, idxMedian._2.asInstanceOf[Double])
        case _ => throw new IllegalArgumentException(
          s"Unsupported value type $colType of column ${cols(idxMedian._1)}.")
      }
    })

    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for ((idx, fillV) <- idxMedians) {
        if (row.isNullAt(idx)) {
          origin.update(idx, fillV)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = df.sparkSession
    spark.createDataFrame(dfUpdated, df.schema)
  }

  /* ---- Stat Operator ---- */
  def median(df: DataFrame, columns: JList[String] = null, relativeError: Double = 0.00001):
  DataFrame = {
    val cols = if (columns == null) {
      df.columns.filter(column => Utils.checkColumnNumeric(df, column))
    } else {
      columns.asScala.toArray
    }

    Utils.getIndex(df, cols)  // checks if `columns` exist in `df`
    val medians = Utils.getMedian(df, cols, relativeError)
    val medians_data = (cols zip medians).map(cm => Row.fromSeq(Array(cm._1, cm._2)))
    val spark = df.sparkSession
    val schema = StructType(Array(
      StructField("column", StringType, nullable = true),
      StructField("median", DoubleType, nullable = true)
    ))
    spark.createDataFrame(spark.sparkContext.parallelize(medians_data), schema)
  }

  def ordinalShufflePartition(df: DataFrame): DataFrame = {
    val shuffledDF = df.withColumn("ordinal", (rand() * pow(2, 52)).cast(LongType))
      .sortWithinPartitions(col("ordinal")).drop(col("ordinal"))
    shuffledDF
  }

  def dfWriteParquet(df: DataFrame, path: String, mode: String = "overwrite"): Unit = {
    df.write.mode(mode).parquet(path)
  }
}
