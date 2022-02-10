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
 
package com.intel.analytics.bigdl.dllib.examples.nnframes.xgboost

import org.apache.spark.sql.{SparkSession, Row}

class Task extends Serializable{
  def rowToLibsvm(row: Row): String = {
    0 until row.length flatMap {
      case 0 => Some(row(0).toString)
      case i if row(i) == null => Some("-999")
      case i => Some( (if (i < 14) row(i) else java.lang.Long.parseLong(row(i).toString, 16)).toString )
    } mkString " "
  }
}

object preprocessData {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder().getOrCreate()

    val task = new Task()

    val input_path = args(0) // path to iris.data
    val output_path = args(1) // save to this path

    var df = spark.read.option("header", "false").option("inferSchema", "true").option("delimiter", "\t").csv(input_path)

    df.rdd.map(task.rowToLibsvm).saveAsTextFile(output_path)

    spark.stop()
  }
}
