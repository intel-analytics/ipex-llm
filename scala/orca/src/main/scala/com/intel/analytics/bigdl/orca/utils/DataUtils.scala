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

package com.intel.analytics.bigdl.orca.utils

import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.types.{ArrayType, IntegerType}
import org.apache.spark.sql.{DataFrame, Row}

import scala.collection.mutable
import scala.collection.mutable.WrappedArray
import scala.util.Random

object DataUtils {
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

  def getIndex(df: DataFrame, columns: Array[String]): Array[Int] = {
    columns.map(col_n => {
      val idx = df.columns.indexOf(col_n)
      if (idx == -1) {
        throw new IllegalArgumentException(s"The column name $col_n does not exist")
      }
      idx
    })
  }
}

