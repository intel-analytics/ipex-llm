/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.friesian.serving.utils.ranking

import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{T, Table}
import org.apache.log4j.Logger
import org.apache.spark.sql.SparkSession

object RankingUtils {
  val logger: Logger = Logger.getLogger(getClass)

  def loadParquetAndConvert(spark: SparkSession, dataDir: String): Table = {
    val df = spark.read.parquet(dataDir)
    val wideCols = Array("engaged_with_user_is_verified", "enaging_user_is_verified")
    val crossCols = Array("present_media_language")
    val indicatorCols = Array("present_media", "tweet_type", "language",
      "engaged_with_user_follower_count",
      "engaged_with_user_following_count",
      "enaging_user_follower_count",
      "enaging_user_following_count")
    val contCols = Array("len_hashtags", "len_domains", "len_links")
    val wndCols = wideCols ++ crossCols ++ indicatorCols ++ contCols
    print(wndCols.mkString(", "))
    val data = df.rdd.map(row => {
      wndCols.map(colName => row.getAs[Number](colName))
    }).collect()
    println()
    print(s"Total number of records: ${data.length}")
    println()
    val dataTensorList = (0 until 13).map(idx => {
      Tensor[Float](T.seq(data.map(d => {
        d(idx)
      })))
    }).toArray
    val input = T.array(dataTensorList)
    input
  }

  def activityToList(result: Activity): List[Float] = {
    result.toTensor[Float].squeeze(2).toArray().toList
  }
}
