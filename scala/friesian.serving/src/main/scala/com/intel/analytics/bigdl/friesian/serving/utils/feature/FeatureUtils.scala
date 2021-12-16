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

package com.intel.analytics.bigdl.friesian.serving.utils.feature

import java.util.{Base64, List => JList}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.T
import com.intel.analytics.bigdl.friesian.serving.feature.utils.RedisUtils
import com.intel.analytics.bigdl.friesian.serving.utils.{EncodeUtils, Utils}
import com.intel.analytics.bigdl.orca.inference.InferenceModel
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto._
import EncodeUtils.objToBytes
import org.apache.log4j.Logger
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.{Row, SparkSession}

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

object FeatureUtils {
  val logger: Logger = Logger.getLogger(getClass)

  def loadUserItemFeaturesRDD(spark: SparkSession): Unit = {
    assert(Utils.helper.initialUserDataPath != null ||
      Utils.helper.initialItemDataPath != null, "initialUserDataPath or " +
      "initialItemDataPath should be provided if loadInitialData is true")
    val redis = RedisUtils.getInstance(Utils.helper.redisPoolMaxTotal)
    if (Utils.helper.initialUserDataPath != null) {
      assert(Utils.helper.userIDColumn != null)
      assert(Utils.helper.userFeatureColArr != null)
      logger.info("Start inserting user features...")
      val colNames = Utils.helper.userFeatureColArr.mkString(",")
      redis.setSchema("user", colNames)
      val userFeatureColumns = Utils.helper.userIDColumn +: Utils.helper.userFeatureColArr
      divideFileAndLoad(spark, Utils.helper.initialUserDataPath, userFeatureColumns,
        "user")
    }

    if (Utils.helper.initialItemDataPath != null) {
      assert(Utils.helper.itemIDColumn != null)
      assert(Utils.helper.itemFeatureColArr != null)
      logger.info("Start inserting item features...")
      val colNames = Utils.helper.itemFeatureColArr.mkString(",")
      redis.setSchema("item", colNames)
      val itemFeatureColumns = Utils.helper.itemIDColumn +: Utils.helper.itemFeatureColArr
      divideFileAndLoad(spark, Utils.helper.initialItemDataPath, itemFeatureColumns,
        "item")
    }
    logger.info(s"Insert finished")
  }

  def divideFileAndLoad(spark: SparkSession, dataDir: String, featureCols: Array[String],
                        keyPrefix: String): Unit = {
    var totalCnt: Long = 0
    val readList = Utils.getListOfFiles(dataDir)
    val start = System.currentTimeMillis()
    for (parquetFiles <- readList) {
      var df = spark.read.parquet(parquetFiles: _*)
      df = df.select(featureCols.map(col): _*).distinct()
      val cnt = df.count()
      totalCnt = totalCnt + cnt
      logger.info(s"Load ${cnt} features into redis.")
      val featureRDD = df.rdd.map(row => {
        encodeRow(row)
      })
      featureRDD.foreachPartition { partition =>
        if (partition.nonEmpty) {
          val redis = RedisUtils.getInstance(Utils.helper.redisPoolMaxTotal)
          redis.Mset(keyPrefix, partition.toList.asJava)
        }
      }
    }
    val end = System.currentTimeMillis()
    logger.info(s"Insert ${totalCnt} features into redis, takes: ${(end - start) / 1000}s")
  }

  def encodeRow(row: Row): JList[String] = {
    val id = row.get(0).toString
    val rowArr = row.toSeq.drop(1).toArray
    val encodedValue = java.util.Base64.getEncoder.encodeToString(objToBytes(rowArr))
    List(id, encodedValue).asJava
  }

  @Deprecated
  def encodeRowWithCols(row: Row, cols: Array[String]): JList[String] = {
    val rowSeq = row.toSeq
    val id = rowSeq.head.toString
    val colValueMap = (cols zip rowSeq).toMap
    val encodedValue = java.util.Base64.getEncoder.encodeToString(
      objToBytes(colValueMap))
    List(id, encodedValue).asJava
  }

  def doPredict(ids: IDs, model: InferenceModel): JList[String] = {
    val idsScala = ids.getIDList.asScala
    val input = Tensor[Float](T.seq(idsScala))
    val result: Tensor[Float] = model.doPredict(input).toTensor
    idsScala.indices.map(idx => {
      val dTensor: Tensor[Float] = result.select(1, idx + 1)
      java.util.Base64.getEncoder.encodeToString(
        objToBytes(dTensor))
    }).toList.asJava
  }

  def predictFeatures(features: Features, model: InferenceModel, featureColumns: Array[String]):
  JList[String] = {
    val featureArr = FeatureUtils.getFeatures(features)
    val featureArrNotNull = featureArr.filter(_ != null)
    if (featureArrNotNull.length == 0) {
      throw new Exception("Cannot find target user/item in redis.")
    }
    val featureListIDArr = featureArrNotNull.map(f => {
      val fMap = f.asInstanceOf[Map[String, AnyRef]]
      val fList = featureColumns.map(colName => {
        fMap.getOrElse(colName, -1)
      })
      fList
    })

    val tensorArray = ArrayBuffer[Tensor[Float]]()

    featureColumns.indices.foreach(idx => {
      var singleDim = true
      val converted = featureListIDArr.map(singleFeature => {
        singleFeature(idx) match {
          case d: Int => Array(d.toFloat)
          case d: Float => Array(d)
          case d: Long => Array(d.toFloat)
          case d: mutable.WrappedArray[AnyRef] =>
            singleDim = false
            var isNumber = true
            if (d.nonEmpty) {
              if (d(0).isInstanceOf[String]) {
                isNumber = false
              }
            }
            if (isNumber) {
              d.toArray.map(_.asInstanceOf[Number].floatValue())
            } else {
              d.toArray.map(_.asInstanceOf[String].toFloat)
            }
          case _ => throw new IllegalArgumentException("")
        }
      })
      val inputTensor = if (singleDim) {
        Tensor[Float](converted.flatten, Array(converted.length))
      } else {
        Tensor[Float](converted.flatten, Array(converted.length, converted(0).length))
      }
      tensorArray.append(inputTensor)
    })
    val inputFeature = T.array(tensorArray.toArray)
    val result: Tensor[Float] = model.doPredict(inputFeature).toTensor
    var notNullIdx = 1
    featureArr.map(f => {
      if (f == null) {
        ""
      } else {
        val b64Str = java.util.Base64.getEncoder.encodeToString(
          objToBytes(result.select(1, notNullIdx)))
        notNullIdx = notNullIdx + 1
        b64Str
      }
    }).toList.asJava
  }

  def getFeatures(features: Features): Array[Array[Any]] = {
    val b64Features = features.getB64FeatureList.asScala
    b64Features.map(feature => {
      if (feature == "") {
        null
      } else {
        EncodeUtils.bytesToObj(Base64.getDecoder.decode(feature)).asInstanceOf[Array[Any]]
      }
    }).toArray
  }
}
