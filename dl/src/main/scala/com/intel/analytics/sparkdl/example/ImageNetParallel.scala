/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.sparkdl.example

import com.intel.analytics.sparkdl.example.ImageNetUtils._
import com.intel.analytics.sparkdl.example.Utils._
import com.intel.analytics.sparkdl.models.imagenet.GoogleNet_v2_NoAuxClassifier
import com.intel.analytics.sparkdl.nn._
import com.intel.analytics.sparkdl.optim._
import com.intel.analytics.sparkdl.optim.SGD
import com.intel.analytics.sparkdl.optim.SGD.{EpochStep, EpochSchedule, Poly, Regime}
import com.intel.analytics.sparkdl.ps.{AllReduceParameterManager, OneReduceParameterManager}
import com.intel.analytics.sparkdl.tensor._
import com.intel.analytics.sparkdl.utils.T
import org.apache.hadoop.io.Text
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object ImageNetParallel {
  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.sparkdl.optim").setLevel(Level.INFO)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {
      train(params)
    }
    }.getOrElse {
      sys.exit(1)
    }
  }

  private def train(params: Params) = {
    val trainFiles = params.folder + "/train"
    val testFiles = params.folder + "/validation"
    val labelFile = params.labelsFile
    val partitionNum = params.partitionNum
    val optType = params.masterOptM
    val netType = params.net
    val classNum = params.classNum
    val conf = new SparkConf().setAppName(s"ImageNet class[Float]: ${params.classNum} " +
      s"Parallelism: ${params.parallelism.toString}, partition : ${params.partitionNum}, " +
      s"masterConfig: ${params.masterConfig}, workerConfig: ${params.workerConfig}")

    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_BLOCKTIME", "0")
    conf.setExecutorEnv("OMP_WAIT_POLICY", "passive")
    conf.setExecutorEnv("OMP_NUM_THREADS", "1")
    conf.set("spark.task.maxFailures", "1")
    conf.set("spark.shuffle.blockTransferService", "nio")
    conf.set("spark.akka.frameSize", "10") // akka networking speed is slow
    val sc = new SparkContext(conf)
    val cropImage = params.crop
    println(s"crop $cropImage")

    val labelsMap = getLabels(labelFile)
    val trainData = if (cropImage) {
      loadCroppedData(trainFiles, sc, labelsMap, classNum + 0.5)
    } else {
      loadData(trainFiles, sc, labelsMap, classNum + 0.5)
    }
    println(s"Count train data...")
    val totalSize = trainData.count()
    println(s"train data size is ${totalSize}")
    println("Estimate mean and std...")
    val sum = computeMean(trainData, params.crop)
    val meanR = sum._1.toFloat / totalSize
    val meanG = sum._2.toFloat / totalSize
    val meanB = sum._3.toFloat / totalSize
    val mean = (meanR, meanG, meanB)
    println(s"meanR is $meanR meanG is $meanG meanB is $meanB")
    val varSum = computeVar(trainData, meanR, meanG, meanB, params.crop)
    val varR = varSum._1.toFloat / totalSize
    val varG = varSum._2.toFloat / totalSize
    val varB = varSum._3.toFloat / totalSize
    val std = (varR, varG, varB)
    println(s"varR is $varR varG is $varG varB is $varB")

    val criterion = new ClassNLLCriterion[Float]()
    val model = netType match {
      case "alexnet" => AlexNet.getModel[Float](classNum)
      case "googlenet" => GoogleNet.getModelCaffe[Float](classNum)
      case "googlenet_v2" => GoogleNet_v2_NoAuxClassifier[Float](classNum)
    }
    println(model)

    val driverConfig = params.masterConfig.clone()
    val workerConfig = params.workerConfig.clone()
    workerConfig("profile") = true

    //driverConfig("learningRateSchedule") = Poly(0.5, 62000)
    //driverConfig("learningRateSchedule") = Poly(0.5, 90000)
    driverConfig("learningRateSchedule") = EpochStep(2, 0.96)

    val croppedData = if (cropImage) {
      loadCroppedData(trainFiles, sc, labelsMap, classNum + 0.5).coalesce(partitionNum, true)
    } else {
      loadData(trainFiles, sc, labelsMap, classNum + 0.5).coalesce(partitionNum, true)
    }

    croppedData.setName("Cropped Train Data").cache()
    println("Cache cropped data...")
    croppedData.count()

    val parameter = model.getParameters()._1
    val metrics = new Metrics
    val dataSets = new MultiThreadShuffleBatchDataSet[(Float, Array[Byte]), Float](croppedData,
    //val dataSets = new ShuffleBatchDataSet[(Float, Array[Byte]), Float](croppedData,
      if (cropImage) toTensorWithoutCrop(mean, std) else toTensor(mean, std),
      params.workerConfig[Int]("stack"), params.workerConfig[Int]("batch"))
    val pm = if (params.pmType == "allreduce") {
      new AllReduceParameterManager[Float](parameter, dataSets.partitions(), metrics)
    } else if (params.pmType == "onereduce") {
      new OneReduceParameterManager[Float](parameter, dataSets.partitions(), metrics)
    } else {
      throw new IllegalArgumentException()
    }

    val validation = croppedData.zipPartitions((if (cropImage) {
      loadCroppedData(testFiles, sc, labelsMap, classNum + 0.5)
    } else {
      loadData(testFiles, sc, labelsMap, classNum + 0.5)
    }).coalesce(partitionNum, true))((train, test) => test).cache()
    validation.setName("Validation Data")
    println(s"load validation data")
    val size = validation.count()
    println(s"validation data size is ${size}")

    val testDataSets = new ShuffleBatchDataSet[(Float, Array[Byte]), Float](validation,
      if (cropImage) toTensorWithoutCrop(mean, std) else toTensor(mean, std), 4, 4)

    val optimizer = new BetterGradAggEpochOptimizer[Float](model, criterion,
    //val optimizer = new GradAggEpochOptimizer[Float](model, criterion,
      getOptimMethodFloat(params.masterOptM),
      pm, dataSets, metrics, driverConfig)
    optimizer.addEvaluation("top1", EvaluateMethods.calcAccuracy)
    optimizer.addEvaluation("top5", EvaluateMethods.calcTop5Accuracy)
    optimizer.setTestDataSet(testDataSets)
    optimizer.setMaxEpoch(100)

    conf.set("spark.task.cpus", params.parallelism.toString)
    optimizer.optimize()
  }

  def toTensor(mean: (Float, Float, Float),
    std: (Float, Float, Float))(images: Seq[(Float, Array[Byte])], input: Tensor[Float],
    target: Tensor[Float]): (Tensor[Float], Tensor[Float]) = {
    val size = images.size
    val featureShape = Array(3, 224, 224)
    val featureSize = featureShape.product
    input.resize(Array(size) ++ featureShape)
    target.resize(Array(size))
    val features = input.storage().array()
    val targets = target.storage().array()
    var i = 0
    while (i < size) {
      val (label, data) = images(i)
      cropFloat(data, input.size(3), input.size(4), mean, std, features, i * featureSize)
      targets(i) = label
      i += 1
    }
    (input, target)
  }

  def toTensorWithoutCrop(mean: (Float, Float, Float),
    std: (Float, Float, Float))(images: Seq[(Float, Array[Byte])],
    input: Tensor[Float], target: Tensor[Float]): (Tensor[Float], Tensor[Float]) = {
    val size = images.size
    val featureShape = Array(3, 224, 224)
    val featureSize = featureShape.product
    input.resize(Array(size) ++ featureShape)
    target.resize(Array(size))
    val features = input.storage().array()
    val targets = target.storage().array()
    var i = 0
    while (i < size) {
      val (label, data) = images(i)
      normalizeFloat(data, 224 * 224, mean, std, features, i * featureSize)
      targets(i) = label
      i += 1
    }
    (input, target)
  }

  def computeMean(data: RDD[(Float, Array[Byte])], cropped: Boolean): (Double, Double, Double) = {
    data.map(d => ImageNetUtils.computeMean(d._2, if (cropped) 0 else 8)).
      reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
  }

  def computeVar(data: RDD[(Float, Array[Byte])], meanR: Double, meanG: Double, meanB: Double,
    cropped: Boolean): (Double, Double, Double) = {
    data.map(d => ImageNetUtils.computeVar(d._2, meanR, meanG, meanB, if (cropped) 0 else 8)).
      reduce((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3))
  }

  def loadData(url: String, sc: SparkContext, labelMap: Map[String, Double],
    classes: Double): RDD[(Float, Array[Byte])] = {
    val labelMapBroadCast = sc.broadcast(labelMap)
    sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
      // extract label
      val localLabelMap = labelMapBroadCast.value
      val label = image._1.toString.split("_")(0)
      require(label.matches("n\\d\\d\\d\\d\\d\\d\\d\\d"), "Invalid label format")
      (localLabelMap(label), image)
    }).filter(_._1 < classes).map(labelAndimage => {
      val data = labelAndimage._2._2.getBytes
      // Hadoop Reader always resuse the same key value reader so let's make a copy of it
      val newData = new Array[Byte](data.length)
      System.arraycopy(data, 0, newData, 0, data.length)

      (labelAndimage._1.toFloat, newData)
    })
  }

  def loadCroppedData(url: String, sc: SparkContext, labelMap: Map[String, Double],
    classes: Double)
  : RDD[(Float, Array[Byte])] = {
    val labelMapBroadCast = sc.broadcast(labelMap)
    sc.sequenceFile(url, classOf[Text], classOf[Text]).map(image => {
      // extract label
      val localLabelMap = labelMapBroadCast.value
      val label = image._1.toString.split("_")(0)
      require(label.matches("n\\d\\d\\d\\d\\d\\d\\d\\d"), "Invalid label format")
      (localLabelMap(label), image)
    }).filter(_._1 < classes).map(labelAndimage => {
      val data = labelAndimage._2._2.getBytes
      // Hadoop Reader always resuse the same key value reader so let's make a copy of it
      val result = new Array[Byte](3 * 224 * 224)
      crop(data, 224, 224, result)

      (labelAndimage._1.toFloat, result)
    })
  }
}
