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

package com.intel.analytics.bigdl.orca.python

import com.intel.analytics.bigdl.orca.inference.InferenceModel
import org.apache.spark.api.java.{JavaRDD, JavaSparkContext}
import java.util.{List => JList}

import com.intel.analytics.bigdl.dllib.tensor.TensorNumericMath.TensorNumeric
import com.intel.analytics.bigdl.dllib.common.PythonZoo

import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import org.apache.spark.TaskContext
import org.apache.spark.sql._
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.functions.{col, rand, row_number, spark_partition_id, udf, log => sqllog}

object PythonOrca {

  def ofFloat(): PythonOrca[Float] = new PythonOrca[Float]()

  def ofDouble(): PythonOrca[Double] = new PythonOrca[Double]()
}

class PythonOrca[T: ClassTag](implicit ev: TensorNumeric[T]) extends PythonZoo[T] {
  def inferenceModelDistriPredict(model: InferenceModel, sc: JavaSparkContext,
                                  inputs: JavaRDD[JList[com.intel.analytics.bigdl.dllib.
                                  utils.python.api.JTensor]],
                                  inputIsTable: Boolean): JavaRDD[JList[Object]] = {
    val broadcastModel = sc.broadcast(model)
    inputs.rdd.mapPartitions(partition => {
      val localModel = broadcastModel.value
      partition.map(inputs => {
        val inputActivity = jTensorsToActivity(inputs, inputIsTable)
        val outputActivity = localModel.doPredict(inputActivity)
        activityToList(outputActivity)
      })
    })
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
      val count_list: Array[(Int, Int)] = df_col_filtered.rdd.mapPartitions(getPartitionSize)
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

  def getPartitionSize(rows: Iterator[Row]): Iterator[(Int, Int)] = {
    if (rows.isEmpty) {
      Array[(Int, Int)]().iterator
    } else {
      val part_id = TaskContext.get().partitionId()
      Array(Tuple2(part_id, rows.size)).iterator
    }
  }
}
