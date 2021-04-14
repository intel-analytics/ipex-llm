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

import org.apache.spark.TaskContext
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

  def getPartitionSize(rows: Iterator[Row]): Iterator[(Int, Int)] = {
    if (rows.isEmpty) {
      Array[(Int, Int)]().iterator
    } else {
      val part_id = TaskContext.get().partitionId()
      Array(Tuple2(part_id, rows.size)).iterator
    }
  }
}
