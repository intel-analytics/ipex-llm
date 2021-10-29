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

package com.intel.analytics.bigdl.ppml.vfl.example.logisticregression

import com.intel.analytics.bigdl.dllib.feature.dataset.{DataSet, MiniBatch, Sample, SampleToMiniBatch}
import com.intel.analytics.bigdl.dllib.tensor.Tensor
import com.intel.analytics.bigdl.ppml.FLClient
import com.intel.analytics.bigdl.ppml.psi.test.TestUtils
import com.intel.analytics.bigdl.ppml.vfl.LogisticRegression
import com.intel.analytics.bigdl.ppml.vfl.example.ExampleUtils
import com.intel.analytics.bigdl.{DataSet, Module}
import org.apache.log4j.Logger
import scopt.OptionParser

import scala.io.Source
import collection.JavaConverters._
import collection.JavaConversions._

/**
 * A two process example to simulate 2 nodes Vfl of a Neural Network
 * This example will start a FLServer first, to provide PSI algorithm
 * and store parameters as Parameter Server
 */
object VflLogisticRegression {
  var dataSet: Array[Array[Float]] = null
  var trainData: DataSet[MiniBatch[Float]] = null
  var valData: DataSet[MiniBatch[Float]] = null
  var featureNum: Int = _
  var model: Module[Float] = null
  var flClient: FLClient = new FLClient()
  var batchSize: Int = 0
  val logger = Logger.getLogger(getClass)

  protected var hashedKeyPairs: Map[String, String] = null


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

    /**
     * Split data into train and validation set
     */
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
    val trainDataset = DataSet.array(samples) ->
      SampleToMiniBatch(batchSize, parallelizing = false)
    //TODO: Find a better dataset has val dataset.
    val valDataSet = DataSet.array(samples) ->
      SampleToMiniBatch(batchSize, parallelizing = false)
    (trainDataset, valDataSet)
  }


  def uploadKeys(keys: Array[String]): Map[String, String] = {
    val salt = flClient.psiStub.getSalt
    logger.debug("Client get Salt=" + salt)
    val hashedKeys = TestUtils.parallelToSHAHexString(keys, salt)
    val hashedKeyPairs = hashedKeys.zip(keys).toMap
    // Hash(IDs, salt) into hashed IDs
    logger.debug("HashedIDs Size = " + hashedKeys.size)
    flClient.psiStub.uploadSet(hashedKeys.toList.asJava)
    hashedKeyPairs

  }
  def getIntersectionKeys(): Array[String] = {
    require(null != hashedKeyPairs, "no hashed key pairs found, have you upload keys?")
    // TODO: just download
    var maxWait = 20
    var intersection: java.util.List[String] = null
    while (maxWait > 0) {
      intersection = flClient.psiStub.downloadIntersection()
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


  def main(args: Array[String]): Unit = {
    case class Params(dataPath: String = null,
                      rowKeyName: String = null,
                      learningRate: Float = 0.005f,
                      batchSize: Int = 4)
    val parser: OptionParser[Params] = new OptionParser[Params]("VFL Logistic Regression") {
      opt[String]('d', "dataPath")
        .text("data path to load")
        .action((x, params) => params.copy(dataPath = x))
        .required()
      opt[String]('r', "rowKeyName")
        .text("row key name of data")
        .action((x, params) => params.copy(rowKeyName = x))
        .required()
      opt[String]('l', "learningRate")
        .text("learning rate of training")
        .action((x, params) => params.copy(learningRate = x.toFloat))
      opt[String]('b', "batchSize")
        .text("batchsize of training")
        .action((x, params) => params.copy(batchSize = x.toInt))
    }
    val argv = parser.parse(args, Params()).head
    // load args and get data
    val dataPath = argv.dataPath
    val rowKeyName = argv.rowKeyName
    val learningRate = argv.learningRate
    batchSize = argv.batchSize
    getData(dataPath, rowKeyName)

    // create LogisticRegression object to train the model
    val lr = new LogisticRegression(featureNum, learningRate)
    lr.fit(trainData, valData)
    lr.evaluate()
  }

}
