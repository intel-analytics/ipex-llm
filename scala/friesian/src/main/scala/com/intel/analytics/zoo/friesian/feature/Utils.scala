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

package com.intel.analytics.zoo.friesian.feature

import java.util
import java.util.{List => JList}

import org.apache.spark.TaskContext
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

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
      if(idx == -1) {
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
}
