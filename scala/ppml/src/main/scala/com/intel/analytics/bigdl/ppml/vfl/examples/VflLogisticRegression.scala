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

package com.intel.analytics.bigdl.ppml.vfl.examples


import com.intel.analytics.bigdl.{DataSet, Module}
import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, MiniBatch, Sample}
import com.intel.analytics.bigdl.dllib.nn.{Linear, Sequential}
import com.intel.analytics.bigdl.dllib.optim.Adam
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.dllib.utils.RandomGenerator.RNG
import com.intel.analytics.bigdl.ppml.psi.test.TestUtils
import com.intel.analytics.bigdl.ppml.{FLClient, FLServer}
import com.intel.analytics.bigdl.ppml.vfl.VflEstimator
import com.intel.analytics.bigdl.ppml.vfl.utils.SampleToMiniBatch
import org.apache.log4j.Logger

import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.io.Source

/**
 * A two process example to simulate 2 nodes VFL of a Neural Network
 * This example will start a FLServer first, to provide PSI algorithm
 * and store parameters as Parameter Server
 */
object VflLogisticRegression {
  val logger = Logger.getLogger(this.getClass)

  /**
   * Start local trainers
   */
  def start(dataPath: String,
            rowKeyName: String,
            batchSize: Int,
            learningRate: Float): Unit = {
    val localVFLTrainer = new LocalVflTrainer(batchSize, learningRate)
    localVFLTrainer.getData(dataPath, rowKeyName)
    localVFLTrainer.getSplitedTrainEvalData()
    localVFLTrainer.model =
      Sequential[Float]().add(Linear(localVFLTrainer.featureNum, 1))
    localVFLTrainer.train()
    localVFLTrainer.evaluate()
  }

  def main(args: Array[String]): Unit = {
    // load args
    val dataPath = args(0)
    val worker = args(1).toInt
    val batchSize = args(2).toInt
    val learningRate = args(3).toFloat
    val rowKeyName = args(4)
    start(dataPath, rowKeyName, batchSize, learningRate)
  }

}
class LocalVflTrainer(batchSize: Int, learningRate: Float) {
  val flClient = new FLClient()
  var dataSet: Array[Array[Float]] = null
  var trainData: DataSet[MiniBatch[Float]] = null
  var valData: DataSet[MiniBatch[Float]] = null
  var headers: Array[String] = null
  var featureNum: Int = _
  var model: Module[Float] = null
  val logger = Logger.getLogger(getClass)
  protected var hashedKeyPairs: Map[String, String] = null
  val estimator = VflEstimator(model, new Adam(learningRate))
  def train() = {
    estimator.train(30, trainData.toLocal(), valData.toLocal())
  }
  def evaluate() = {
    println(model.getParametersTable())
    estimator.getEvaluateResults().foreach{r =>
      println(r._1 + ":" + r._2.mkString(","))
    }
  }
  def getData(dataPath: String, rowKeyName: String) = {
    // load data from dataset and preprocess
    val sources = Source.fromFile(dataPath, "utf-8").getLines()
    val headers = sources.next().split(",").map(_.trim)
    println(headers.mkString(","))
    val rowKeyIndex = headers.indexOf(rowKeyName)
    require(rowKeyIndex != -1, s"couldn't find ${rowKeyName} in headers(${headers.mkString(", ")})")
    val data = sources.toArray.map{line =>
      val lines = line.split(",").map(_.trim())
      (lines(rowKeyIndex), (lines.take(rowKeyIndex) ++ lines.drop(rowKeyIndex + 1)).map(_.toFloat))
    }.toMap
    val ids = data.keys.toArray
    hashedKeyPairs = uploadKeys(ids)
    val intersections = getIntersectionKeys()
    dataSet = intersections.map{id =>
      data(id)
    }
  }
  def getSplitedTrainEvalData() = {
    val samples = if (headers.last == "Outcome") {
      println("hasLabel")
      featureNum = headers.length - 2
      (0 until featureNum).foreach(i => ExampleUtils.minMaxNormalize(dataSet, i))
      (dataSet.map{d =>
        val features = Tensor[Float](d.slice(0, featureNum), Array(featureNum))
        val target = Tensor[Float](Array(d(featureNum)), Array(1))
        Sample(features, target)
      })
    } else {
      println("no label")
      featureNum = headers.length - 1
      (0 until featureNum).foreach(i => ExampleUtils.minMaxNormalize(dataSet, i))
      (dataSet.map{d =>
        val features = Tensor[Float](d, Array(featureNum))
        Sample(features)
      })
    }
    val trainDataset = DataSet.array(samples) -> SampleToMiniBatch(batchSize)
    //TODO: Find a better dataset has val dataset.
    val valDataSet = DataSet.array(samples) -> SampleToMiniBatch(batchSize)
    (trainDataset, valDataSet)
  }

  def uploadKeys(keys: Array[String]): Map[String, String] = {
    val salt = flClient.getSalt
    logger.debug("Client get Salt=" + salt)
    val hashedKeys = TestUtils.parallelToSHAHexString(keys, salt)
    val hashedKeyPairs = hashedKeys.zip(keys).toMap
    // Hash(IDs, salt) into hashed IDs
    logger.debug("HashedIDs Size = " + hashedKeys.size)
    flClient.uploadSet(hashedKeys.toList.asJava)
    hashedKeyPairs

  }
  def getIntersectionKeys(): Array[String] = {
    require(null != hashedKeyPairs, "no hashed key pairs found, have you upload keys?")
    // TODO: just download
    var maxWait = 20
    var intersection: java.util.List[String] = null
    while (maxWait > 0) {
      intersection = flClient.downloadIntersection()
      if (intersection == null || intersection.length == 0) {
        logger.info("Wait 1000ms")
        Thread.sleep(1000)
      } else {
        logger.info("Intersection successful. The id(s) in the intersection are: ")
        logger.info(intersection.mkString(", "))
        logger.info("Origin IDs are: ")
        logger.info(intersection.map(hashedKeyPairs(_)).mkString(", "))
        //break
        maxWait = 1
      }
      maxWait -= 1
    }
    intersection.asScala.toArray.map { k =>
      require(hashedKeyPairs.contains(k), "unknown intersection keys, please check psi server.")
      hashedKeyPairs(k)
    }
  }
}