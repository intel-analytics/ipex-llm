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

package com.intel.analytics.bigdl.example

import com.intel.analytics.bigdl.example.MNIST._
import com.intel.analytics.bigdl.example.Utils._
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.ps.OneReduceParameterManager
import org.apache.log4j.{Level, Logger}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object MNISTParallel {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)

    val parser = getParser()

    parser.parse(args, defaultParams).map { params => {
      train(params)
    }
    }.getOrElse {
      sys.exit(1)
    }
  }

  private def train(params: Params) = {
    val folder = params.folder
    val trainData = loadFile(s"$folder/train-images.idx3-ubyte", s"$folder/train-labels.idx1-ubyte")
    val testData = loadFile(s"$folder/t10k-images.idx3-ubyte", s"$folder/t10k-labels.idx1-ubyte")

    val partitionNum = params.partitionNum
    val conf = new SparkConf().setAppName(s"MNIST with Partition $partitionNum")
    if (params.isLocalSpark) {
      conf.setMaster(s"local[${params.partitionNum}]")
    }
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_AFFINITY", "compact,1,0")
    conf.setExecutorEnv("MKL_DYNAMIC", "false")
    conf.setExecutorEnv("OMP_NUM_THREADS", params.parallelism.toString)
    conf.set("spark.task.cpus", params.parallelism.toString)
    conf.set("spark.shuffle.blockTransferService", "nio")

    val sc = new SparkContext(conf)
    val trainRDD = sc.parallelize(trainData).repartition(partitionNum).persist()
    val testRDD = trainRDD.zipPartitions(sc.parallelize(testData).
      repartition(partitionNum))((train, test) => test).persist()
    trainRDD.setName("trainRDD")
    testRDD.setName("testRDD")
    val (mean, std) = computeMeanStd(trainRDD)
    println(s"mean is $mean std is $std")

    val driverConfig = params.masterConfig.clone()

    val workerConfig = params.workerConfig.clone()

    val model = getModule(params.net)
    val parameter = model.getParameters()._1
    val metrics = new Metrics()
    val dataSets = new ShuffleBatchDataSet[Array[Byte], Double](trainRDD,
      toTensor(mean, std), 10, 40, params.workerConfig[Int]("batchnum"))
    val pm = new OneReduceParameterManager[Double](parameter, dataSets.partitions(), metrics)
    val testDataSets = new ShuffleBatchDataSet[Array[Byte], Double](testRDD,
      toTensor(mean, std), 10, 40, params.workerConfig[Int]("batchnum"))
    val optimizer = if (params.distribute == "parallel") {
      new WeightAvgEpochOptimizer[Double](model, new ClassNLLCriterion(),
        getOptimMethod(params.masterOptM), pm, dataSets, metrics, driverConfig)
    } else if (params.distribute == "serial") {
      new GradAggEpochOptimizer[Double](model, new ClassNLLCriterion(),
        getOptimMethod(params.masterOptM), pm, dataSets, metrics, driverConfig)
    } else {
      throw new IllegalArgumentException
    }

    optimizer.setTestInterval(params.valIter)
    optimizer.setPath("./mnist.model")
    optimizer.addEvaluation("top1", EvaluateMethods.calcAccuracy)
    optimizer.setTestDataSet(testDataSets)

    optimizer.optimize()
  }

  def computeMeanStd(data: RDD[Array[Byte]]): (Double, Double) = {
    val count = data.count()
    val total = data.map(img => {
      var i = 1
      var sum = 0.0
      while (i < img.length) {
        sum += (img(i) & 0xff) / 255.0
        i += 1
      }

      sum
    }).reduce(_ + _)

    val mean = total / (count * rowN * colN)

    val stdTotal = data.map(img => {
      var i = 1
      var stdSum = 0.0
      while (i < img.length) {
        val s = (img(i) & 0xff) / 255.0 - mean
        stdSum += s * s
        i += 1
      }

      stdSum
    }).reduce(_ + _)

    val std = math.sqrt(stdTotal / (count * rowN * colN))

    (mean, std)
  }
}
