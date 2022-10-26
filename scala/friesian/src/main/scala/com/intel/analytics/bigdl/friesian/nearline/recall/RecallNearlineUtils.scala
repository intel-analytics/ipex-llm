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

package com.intel.analytics.bigdl.friesian.nearline.recall

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.friesian.nearline.utils.NearlineUtils
import com.intel.analytics.bigdl.friesian.serving.recall.IndexService
import org.apache.logging.log4j.{LogManager, Logger}
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.collection.mutable

object RecallNearlineUtils {
  private val logger: Logger = LogManager.getLogger(classOf[RecallInitializer].getName)

  def loadItemData(indexService: IndexService, dataDir: String): Unit = {
    val spark = SparkSession.builder.getOrCreate
    Log4Error.invalidOperationError(NearlineUtils.helper.itemIDColumn != null,
      "itemIdColumn should be provided if " +
        "loadSavedIndex=false")
    Log4Error.invalidOperationError(NearlineUtils.helper.indexPath != null,
      "indexPath should be provided.")

    val itemFeatureColumns = Array(NearlineUtils.helper.itemIDColumn,
      NearlineUtils.helper.itemEmbeddingColumn)
    val readList = NearlineUtils.getListOfFiles(dataDir)
    val start = System.currentTimeMillis()
    for (parquetFiles <- readList) {
      var df = spark.read.parquet(parquetFiles: _*)
      df = df.select(itemFeatureColumns.map(col): _*).distinct()
      val data = df.rdd.map(row => {
        val id = row.get(0)
        val data = row.get(1)
        (id, data)
      }).collect()

      val resultFlattenArr = if (data.length > 0) {
        val tmpDataArr = data.map(_._2)
        tmpDataArr(0) match {
          case _: DenseVector =>
            tmpDataArr.flatMap(_.asInstanceOf[DenseVector].toArray.map(_.toFloat))
          case _: mutable.WrappedArray[Any] =>
            tmpDataArr.flatMap(
              _.asInstanceOf[mutable.WrappedArray[Any]].array.map(_.toString.toFloat))
          case _ =>
            Log4Error.invalidInputError(condition = false,
              "Recall's initialData parquet files contain unsupported Dtype. ",
              "Make sure ID's type is int and ebd's type is DenseVector or array.")
            new Array[Float](0)
        }
      } else {
        new Array[Float](0)
      }
      val idList = data.map(_._1).map(_.toString.toInt)
      indexService.addWithIds(resultFlattenArr, idList)
    }
    val end = System.currentTimeMillis()
    logger.info(s"Building index takes: ${(end - start) / 1000}s")
    indexService.save(NearlineUtils.helper.indexPath)
  }
}
