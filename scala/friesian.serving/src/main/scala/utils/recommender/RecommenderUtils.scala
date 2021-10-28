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

package utils.recommender

import java.util
import java.util.Base64
import com.intel.analytics.bigdl.dllib.nn.abstractnn.Activity
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.{T, Table}
import com.intel.analytics.bigdl.friesian.serving.utils.EncodeUtils
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.feature.FeatureProto.Features
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingGrpc.RankingBlockingStub
import com.intel.analytics.bigdl.friesian.serving.grpc.generated.ranking.RankingProto.{Content, Prediction}
import io.grpc.StatusRuntimeException
import org.apache.log4j.Logger
import org.apache.spark.ml.linalg.DenseVector
import org.apache.spark.sql.SparkSession
import utils.Utils
import utils.feature.FeatureUtils

import scala.collection.JavaConverters._
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

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
      val featureList = idxArr.map(idx => originFeatureList(idx))
      (item._1, featureList)
    })
    val itemIDArr = modelFeatureItemIdArr.map(_._1)
    val userItemFeatureArr = modelFeatureItemIdArr.map(_._2)
    val batchedFeatureArr = userItemFeatureArr.sliding(batchSizeUse, batchSizeUse).toArray
    val batchedActivityList = batchedFeatureArr.map(featureArr => {
      val tensorArray = ArrayBuffer[Tensor[Float]]()
      inferenceColumns.indices.foreach(idx => {
        var singleDim = true
        val converted = featureArr.map(singleFeature => {
          // TODO: null
          singleFeature(idx) match {
            case d: Int => Array(d.toFloat)
            case d: Float => Array(d)
            case d: Long => Array(d.toFloat)
            case d: mutable.WrappedArray[AnyRef] =>
              singleDim = false
              d.toArray.map(_.asInstanceOf[Number].floatValue())
            case d => throw new IllegalArgumentException(s"Illegal input: ${d}")
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
      T.array(tensorArray.toArray)
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
      val resultStrArr = resultStr.replaceAll("\\[", "").dropRight(2).split("\\],")
      resultStrArr.map(a => {
        a.split(",")(0).toFloat
      })
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
