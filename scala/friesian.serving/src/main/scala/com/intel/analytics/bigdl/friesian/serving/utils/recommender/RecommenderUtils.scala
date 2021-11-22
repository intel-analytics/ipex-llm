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

package com.intel.analytics.bigdl.friesian.serving.utils.recommender

import java.util
import java.util.Base64
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{T, Table}
import com.intel.analytics.bigdl.friesian.serving.utils.{EncodeUtils, Utils}
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.{Features, IDs}
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc.RankingBlockingStub
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingProto.{Content, Prediction}
import com.intel.analytics.bigdl.friesian.serving.utils.feature.FeatureUtils
import io.grpc.StatusRuntimeException
import org.apache.log4j.Logger
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession

import scala.collection.JavaConverters._
import scala.collection.mutable

object RecommenderUtils {
  val logger: Logger = Logger.getLogger(getClass)

  def featuresToRankingInputSet(userFeatures: Features, itemFeatures: Features, batchSize: Int)
  : (Array[Int], Array[Table]) = {
    val userFeatureArr = FeatureUtils.getFeatures(userFeatures)
    assert(userFeatureArr.length == 1, "userFeatures length should be 1")
    val userSchema = userFeatures.getColNamesList.asScala
    if (userFeatureArr(0) == null) {
      throw new Exception("Cannot find user feature, userid: " + userFeatures.getID(0))
    }
    val userFeature = userFeatureArr(0)
    // TODO: not found update
    val itemSchema = itemFeatures.getColNamesList.asScala
    val itemIDs = itemFeatures.getIDList.asScala.toArray.map(_.intValue())
    val itemFeatureArr = itemIDs.zip(FeatureUtils.getFeatures(itemFeatures))
      .filter(idx => idx._2 != null)
    logger.info("Got item feature: " + itemFeatureArr.length)

    val batchSizeUse = if (batchSize <= 0) {
      itemFeatureArr.length
    } else {
      batchSize
    }
    if (batchSizeUse == 0) {
      throw new Exception("The recommend service got 0 valid item features. Please make sure " +
        "your initial datasets are matched.")
    }
    val inferenceColumns = Utils.helper.inferenceColArr
    val featureSchema = itemSchema.++(userSchema)
    val idxArr = inferenceColumns.map(col => featureSchema.indexOf(col))
    if (idxArr.contains(-1)) {
      throw new Exception("The feature " + inferenceColumns(idxArr.indexOf(-1)) + " doesn't exist" +
        " in features.")
    }

    val modelFeatureItemIdArr = itemFeatureArr.map(item => {
      val itemF = item._2
      val originFeatureList = itemF.++(userFeature)
      //      val featureList = idxArr.map(idx => originFeatureList(idx))
      //      (item._1, featureList)
      (item._1, originFeatureList)
    })
    val itemIDArr = modelFeatureItemIdArr.map(_._1)
    val userItemFeatureArr = modelFeatureItemIdArr.map(_._2)
    val batchedFeatureArr = userItemFeatureArr.sliding(batchSizeUse, batchSizeUse).toArray
    val batchedActivityList = batchedFeatureArr.map(featureArr => {
      val tensorArray = idxArr.map(idx => {
        val features = featureArr.map(feature => feature(idx))
        val dim = Array(features.length)
        features(0) match {
          case _: Int => Tensor[Float](features.map(_.toString.toFloat), dim)
          case _: Float => Tensor[Float](features.map(_.toString.toFloat), dim)
          case _: Long => Tensor[Float](features.map(_.toString.toFloat), dim)
          case _:Double => Tensor[Float](features.map(_.toString.toFloat), dim)
          case _: mutable.WrappedArray[Any] =>
            val arr2d = features.map(a =>
              a.asInstanceOf[mutable.WrappedArray[Any]].array.map(_.toString.toFloat))
            Tensor[Float](arr2d.flatten, dim :+ arr2d(0).length)
          case d => throw new IllegalArgumentException(s"Illegal input: ${d}")
        }
      })
      T.array(tensorArray)
    })
    (itemIDArr, batchedActivityList)
  }

  def doPredictParallel(inputArr: Array[Activity],
                        inferenceStub: RankingBlockingStub): Array[String] = {
    val resultArr = inputArr.indices.toParArray.map(idx => {
      val input = Base64.getEncoder.encodeToString(EncodeUtils.objToBytes(inputArr(idx)))
      val predContent = Content.newBuilder().setEncodedStr(input).build()
      var result: Prediction = null
      try {
        result = inferenceStub.doPredict(predContent)
      } catch {
        case e: StatusRuntimeException => throw e
      }

      result.getPredictStr
    })
    resultArr.toArray
  }

  def getTopK(result: Array[String], itemIDArr: Array[Int], k: Int): (Array[Int], Array[Float]) = {
    val resultArr = result.indices.toParArray.map(idx => {
      val resultStr = result(idx)
      val resultActivity = EncodeUtils.bytesToObj(Base64.getDecoder.decode(resultStr))
        .asInstanceOf[Activity]
      if (resultActivity.isTensor) {
        val tensor = resultActivity.toTensor[Float].squeeze(2)
        try {
          tensor.toArray()
        } catch {
          case _: Exception => throw new Exception("Not supported inference result type, please " +
            "modify method getTopK in RecommendUtils to ensure the ranking result is correct.")
        }
      } else {
        throw new Exception("Not supported inference result type, please modify method getTopK in " +
          "RecommendUtils to ensure the ranking result is correct.")
      }
    }).toArray
    val flattenResult = resultArr.flatten
    val zipped = itemIDArr zip flattenResult
    val sorted = zipped.sortWith(_._2 > _._2).take(k)
    val sortedId = sorted.map(_._1)
    val sortedProb = sorted.map(_._2)
    (sortedId, sortedProb)
  }

  // For wnd validation
  def loadResultParquet(resultPath: String):
  (util.Map[Integer, Integer], util.Map[Integer, java.lang.Float]) = {
    val spark = SparkSession.builder.getOrCreate
    val df = spark.read.parquet(resultPath)
    val userItemMap = collection.mutable.Map[Int, Int]()
    val userPredMap = collection.mutable.Map[Int, Float]()
    df.collect().foreach(row => {
      val userId = row.getInt(0)
      val itemId = row.getInt(1)
      val pred = row.getAs[DenseVector](2).toArray(0).toFloat
      userItemMap.update(userId, itemId)
      userPredMap.update(userId, pred)
    })
    (userItemMap.asJava.asInstanceOf[util.Map[Integer, Integer]], userPredMap.asJava
      .asInstanceOf[util.Map[Integer, java.lang.Float]])
  }
}
