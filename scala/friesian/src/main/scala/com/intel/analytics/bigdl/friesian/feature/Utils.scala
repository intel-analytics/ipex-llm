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

package com.intel.analytics.bigdl.friesian.feature

import java.util
import java.util.{List => JList}

import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, IntegerType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.{immutable, mutable}
import scala.collection.mutable.WrappedArray
import reflect.runtime.universe._
import scala.util.Random

private[friesian] object Utils {
  def fillNaIndex(df: DataFrame, fillVal: Any, columns: Array[Int]): DataFrame = {
    val targetType = fillVal match {
      case _: Double | _: Long | _: Int => "numeric"
      case _: String => "string"
      case _: Boolean => "boolean"
      case _ => throw new IllegalArgumentException(
        s"Unsupported value type ${fillVal.getClass.getName} ($fillVal).")
    }

    val schema = df.schema

    val fillValList = columns.map(idx => {
      val matchAndVal = checkTypeAndCast(schema(idx).dataType.typeName, targetType, fillVal)
      if (!matchAndVal._1) {
        throw new IllegalArgumentException(s"$targetType does not match the type of column " +
          s"${schema(idx).name}")
      }
      matchAndVal._2
    })

    val dfUpdated = df.rdd.map(row => {
      val origin = row.toSeq.toArray
      for ((idx, fillV) <- columns zip fillValList) {
        if (row.isNullAt(idx)) {
          origin.update(idx, fillV)
        }
      }
      Row.fromSeq(origin)
    })

    val spark = df.sparkSession
    spark.createDataFrame(dfUpdated, schema)
  }

  def checkTypeAndCast(schemaType: String, targetType: String, fillVal: Any):
  (Boolean, Any) = {
    if (schemaType == targetType) {
      return (true, fillVal)
    } else if (targetType == "numeric") {
      val fillNum = fillVal.asInstanceOf[Number]
      return schemaType match {
        case "long" => (true, fillNum.longValue)
        case "integer" => (true, fillNum.intValue)
        case "double" => (true, fillNum.doubleValue)
        case _ => (false, fillVal)
      }
    }
    (false, fillVal)
  }

  def castNumeric(value: Any, typeName: String): Any = {
    if (value == null) {
      null
    }
    else {
      if (!value.isInstanceOf[Number]) {
        throw new IllegalArgumentException(s"Unsupported data type ${value.getClass.getName}" +
          s" $value")
      }
      typeName match {
        case "long" => value.asInstanceOf[Number].longValue
        case "integer" => value.asInstanceOf[Number].intValue
        case "double" => value.asInstanceOf[Number].doubleValue
        case _ => throw new IllegalArgumentException(s"The type $typeName is not numeric")
      }
    }
  }

  def isLarger(val1: Any, val2: Any, typeName: String): Boolean = {
    typeName match {
      case "long" => val1.asInstanceOf[Long] > val2.asInstanceOf[Long]
      case "integer" => val1.asInstanceOf[Int] > val2.asInstanceOf[Int]
      case "double" => val1.asInstanceOf[Double] > val2.asInstanceOf[Double]
      case _ => throw new IllegalArgumentException(s"The type $typeName is not numeric")
    }
  }

  def getClipFunc[T](minVal: Any, maxVal: Any, colType: String): T => T = {
    if (minVal == null) {
      (value: T) => {
        if (Utils.isLarger(value, maxVal, colType)) maxVal.asInstanceOf[T]
        else value
      }
    } else if (maxVal == null) {
      (value: T) => {
        if (Utils.isLarger(minVal, value, colType)) minVal.asInstanceOf[T]
        else value
      }
    } else {
      if (Utils.isLarger(minVal, maxVal, colType)) {
        throw new IllegalArgumentException(s"min should be smaller than max but get $minVal >" +
          s" $maxVal")
      }
      (value: T) => {
        if (Utils.isLarger(value, maxVal, colType)) maxVal.asInstanceOf[T]
        else if (Utils.isLarger(minVal, value, colType)) minVal.asInstanceOf[T]
        else value
      }
    }
  }

  def getPartitionSize(rows: Iterator[Row]): Iterator[(Int, Int)] = {
    if (rows.isEmpty) {
      Array[(Int, Int)]().iterator
    } else {
      val part_id = TaskContext.get().partitionId()
      Array(Tuple2(part_id, rows.size)).iterator
    }
  }

  def checkColumnNumeric(df: DataFrame, column: String): Boolean = {
    val typeName = df.schema(df.columns.indexOf(column)).dataType.typeName
    typeName == "long" || typeName == "integer" || typeName == "double"
  }

  def getIndex(df: DataFrame, columns: Array[String]): Array[Int] = {
    columns.map(col_n => {
      val idx = df.columns.indexOf(col_n)
      if (idx == -1) {
        throw new IllegalArgumentException(s"The column name $col_n does not exist")
      }
      idx
    })
  }

  def getMedian(df: DataFrame, columns: Array[String], relativeError: Double = 0.001):
  Array[Any] = {
    // approxQuantile: `Array(0.5)` corresponds to the median; the inner array
    //                 of the return value is either empty or a singleton
    df.stat.approxQuantile(columns, Array(0.5), relativeError).map(
      quantiles => {
        if (quantiles.isEmpty) {
          null
        }
        else {
          quantiles(0)
        }
      })
  }

  def hashBucket(content: Any, bucketSize: Int = 1000, start: Int = 0): Int = {
    return (content.hashCode() % bucketSize + bucketSize) % bucketSize + start
  }

  def maskArr: (Int, mutable.WrappedArray[Any]) => Seq[Int] = {
    (maxLength: Int, history: WrappedArray[Any]) => {
      val n = history.length
      val result = if (maxLength > n) {
        (0 to n - 1).map(_ => 1) ++ (0 to (maxLength - n - 1)).map(_ => 0)
      } else {
        (0 to maxLength - 1).map(_ => 1)
      }
      result
    }
  }

  def padArr[T]: (Int, mutable.WrappedArray[T]) => mutable.Seq[T] = {
    (maxLength: Int, history: WrappedArray[T]) => {
      val n = history.length
      val padValue = castValueFromNum(history(0), 0)
      val pads: mutable.Seq[T] = if (maxLength > n) {
        history ++ (0 to maxLength - n - 1).map(_ => padValue)
      } else {
        history.slice(n - maxLength, n)
      }
      pads
    }
  }

  def padMatrix[T]: (Int, WrappedArray[WrappedArray[T]]) => Seq[Seq[T]] = {
    (maxLength: Int, history: WrappedArray[WrappedArray[T]]) => {
      val n = history.length
      val padValue = castValueFromNum(history(0)(0), 0)
      if (maxLength > n) {
        val hishead = history(0)
        val padArray =
          (0 to maxLength - n - 1).map(_ => (0 to hishead.length - 1).map(_ => padValue))
        history ++ padArray
      } else {
        history.slice(n - maxLength, n)
      }
    }
  }

  def castValueFromNum[T](num: T, value: Int): T = {
    val out: Any = num match {
      case _: Double => value.toDouble
      case _: Int => value
      case _: Float => value.toFloat
      case _: Long => value.toLong
      case _ => throw new IllegalArgumentException(
        s"Unsupported value type ${num.getClass.getName} ($num).")
    }
    out.asInstanceOf[T]
  }

  def addNegtiveItem[T](negNum: Int, itemSize: Int): Int => Seq[(Int, Int)] = {
    val r = new Random()
    (itemId: Int) =>
      (1 to negNum).map(x => {
        var negItem = 0
        do {
          negItem = 1 + r.nextInt(itemSize - 1)
        } while (negItem == itemId)
        negItem
      }).map(x => (x, 0)) ++ Seq((itemId, 1))
  }

  def addNegativeList[T](negNum: Int, itemSize: Int): mutable.WrappedArray[Int] => Seq[Seq[Int]] = {
    val r = new Random()
    (history: WrappedArray[Int]) => {
      val r = new Random()
      val negItemSeq: Seq[Seq[Int]] = (0 to history.length - 1).map(i => {
        (0 to negNum - 1).map(j => {
          var negItem = 0
          do {
            negItem = 1 + r.nextInt(itemSize - 1)
          } while (negItem == history(i))
          negItem
        })
      })
      negItemSeq
    }
  }

  def get1row[T](full_rows: Array[Row], colName: String, index: Int, lowerBound: Int): Seq[Any] = {
    val colValue = full_rows(index).getAs[T](colName)
    val historySeq = full_rows.slice(lowerBound, index).map(row => row.getAs[T](colName))
    Seq(colValue, historySeq)
  }

  def addValueSingleCol(df: DataFrame, colName: String, mapBr: Broadcast[Map[Int, Int]],
                        key: String, value: String): DataFrame = {

    val colTypes = df.schema.fields.filter(x => x.name.equalsIgnoreCase(colName))
    val lookup = mapBr.value
    if(colTypes.length > 0) {
      val colType = colTypes(0)
      val replaceUdf = colType.dataType match {
        case IntegerType => udf((x: Int) => lookup.getOrElse(x, 0))
        case ArrayType(IntegerType, _) =>
          udf((arr: WrappedArray[Int]) => arr.map(x => lookup.getOrElse(x, 0)))
        case ArrayType(ArrayType(IntegerType, _), _) =>
          udf((mat: WrappedArray[WrappedArray[Int]]) =>
            mat.map(arr => arr.map(x => lookup.getOrElse(x, 0))))
        case _ => throw new IllegalArgumentException(
          s"Unsupported data type ${colType.dataType.typeName} " +
            s"of column ${colType.name} in addValueFeatures")
      }

      df.withColumn(colName.replace(key, value), replaceUdf(col(colName)))

    } else {
      df
    }
  }
}
