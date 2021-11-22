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

package com.intel.analytics.bigdl.friesian.serving.utils.recall

import java.util.{List => JList}
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.friesian.serving.recall.RecallService
import com.intel.analytics.bigdl.friesian.serving.utils.Utils
import com.intel.analytics.bigdl.orca.inference.InferenceModel
import org.apache.log4j.Logger
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer


object RecallUtils {
  private val logger = Logger.getLogger(classOf[RecallService].getName)

  def loadItemData(indexService: RecallService, dataDir: String, model:InferenceModel,
                   batchSize: Int = 0): Unit = {
    val spark = SparkSession.builder.getOrCreate
    assert(Utils.helper.itemIDColumn != null, "itemIdColumn should be provided if " +
      "loadSavedIndex=false")
    if (model != null) {
      var df = spark.read.parquet(dataDir)
      val cnt = df.select(Utils.helper.getItemIDColumn).distinct().count()
      logger.info("Total id count: " + cnt)
      val partitionNum: Int = (cnt.toFloat/batchSize).ceil.toInt
      df = df.repartition(partitionNum)
      assert(Utils.helper.itemFeatureColArr != null, "itemFeatureColumns should be provided " +
        "if loadSavedIndex=false and itemModelPath != null")
      val itemFeatureColumns = Utils.helper.itemIDColumn +: Utils.helper.itemFeatureColArr
      df = df.select(itemFeatureColumns.map(col):_*).distinct()
      val result = df.rdd.mapPartitions(rows => {
        val rowArray = rows.toArray
        val idList = rowArray.map(row => {
          row.getInt(0)
        })
        val tensorArray = ArrayBuffer[Tensor[Float]]()
        Utils.helper.itemFeatureColArr.indices.foreach(idx => {
          var singleDim = true
          val converted = rowArray.map(singleFeature => {
            // TODO: null
            singleFeature(idx + 1) match {
              case d: Int => Array(d.toFloat)
              case d: Float => Array(d)
              case d: Long => Array(d.toFloat)
              case d: mutable.WrappedArray[AnyRef] =>
                singleDim = false
                d.toArray.map(_.asInstanceOf[Number].floatValue())
              case _ => throw new IllegalArgumentException("")
            }
          })
          val inputTensor = if (singleDim) {
            Tensor[Float](converted.flatten, Array(converted.length))
          } else {
            // TODO: empty
            Tensor[Float](converted.flatten, Array(converted.length, converted(0).length))
          }
          tensorArray.append(inputTensor)
        })
        val inputFeature = T.array(tensorArray.toArray)
        val result: Tensor[Float] = Utils.helper.itemModel.doPredict(inputFeature).toTensor
        val resultFlattenArr = result.storage().array()
        Array((resultFlattenArr, idList)).iterator
      }).collect()
      result.foreach(partResult => {
        val resultFlattenArr = partResult._1
        val idList = partResult._2
        indexService.addWithIds(resultFlattenArr, idList)
      })
    } else {
      val itemFeatureColumns = Array(Utils.helper.itemIDColumn, "prediction")
      val parquetList = Utils.getListOfFiles(dataDir)
      logger.info(s"ParquetList length: ${parquetList.length}")
      val readList = parquetList.sliding(10, 10).toArray
      val start = System.currentTimeMillis()
      for (parquetFiles <- readList) {
        var df = spark.read.parquet(parquetFiles: _*)
        df = df.select(itemFeatureColumns.map(col): _*).distinct()
        val data = df.rdd.map(row => {
          val id = row.getInt(0)
          val data = row.getAs[DenseVector](1).toArray.map(_.toFloat)
          (id, data)
        }).collect()
        val resultFlattenArr = data.flatMap(_._2)
        val idList = data.map(_._1)
        indexService.addWithIds(resultFlattenArr, idList)
      }
      val end = System.currentTimeMillis()
      logger.info(s"Building index takes: ${(end - start) / 1000}s")
    }
    if (Utils.helper.saveBuiltIndex) {
      indexService.save(Utils.helper.indexPath)
    }
  }

  def constructActivity(data: JList[Any]): Tensor[Float] = {
    Tensor[Float](T.seq(data.asScala.map {
      case d: Int => d.toFloat
      case d: Double => d.toFloat
      case d: Float => d
      case d  =>
        throw new IllegalArgumentException(s"Only numeric values are supported, but got ${d}")
    }))
  }

  def featureObjToFloatArr(feature: Any): Array[Float] = {
    feature match {
      case d: Activity => activityToFloatArr(d)
      case d: Array[Any] =>
        if (d.length != 1) {
          throw new Exception(s"Feature column number should be 1, but got: ${d.length}")
        }
        d(0) match {
          case f: DenseVector => denseVectorToFloatArr(f)
          case f: Array[Float] => f
          case _ => throw new Exception(s"Unsupported user vector type, only Activity, " +
            s"DenseVector and Float[] are supported, but got ${d.getClass.getName}")
        }
      case d => throw new Exception(s"Unsupported user vector type, only Activity, DenseVector " +
        s"and Float[] are supported, but got ${d.getClass.getName}")
    }
  }

  def activityToFloatArr(data: Activity): Array[Float] = {
    val dTensor: Tensor[Float] = data.toTensor
    val result = dTensor.squeeze(1).toArray()
    result
  }

  def denseVectorToFloatArr(data: DenseVector): Array[Float] = {
    data.toArray.map(_.toFloat)
  }
}
