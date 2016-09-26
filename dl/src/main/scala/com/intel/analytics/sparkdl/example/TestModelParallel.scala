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

import com.intel.analytics.sparkdl.example.Utils._
import com.intel.analytics.sparkdl.nn.ClassNLLCriterion
import com.intel.analytics.sparkdl.optim.{GradAggEpochOptimizer, Metrics, ShuffleBatchDataSet}
import com.intel.analytics.sparkdl.ps.{OneReduceParameterManager, AllReduceParameterManager}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}

object TestModelParallel {
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
    val conf = new SparkConf().setAppName(s"Test")
    conf.setExecutorEnv("MKL_DISABLE_FAST_MM", "1")
    conf.setExecutorEnv("KMP_BLOCKTIME", "0")
    conf.setExecutorEnv("OMP_WAIT_POLICY", "passive")
    conf.setExecutorEnv("OMP_NUM_THREADS", s"${params.parallelism}")
    conf.set("spark.task.maxFailures", "1")
    conf.set("spark.shuffle.blockTransferService", "nio")
    conf.set("spark.akka.frameSize", "10") // akka networking speed is slow
    conf.set("spark.task.cpus", params.parallelism.toString)
    val sc = new SparkContext(conf)

    val classNum = 1000
    val netType = params.net
    println("cache data")
    val trainData = sc.parallelize(1 to params.partitionNum * 10000, params.partitionNum).cache()
    trainData.count()
    println("done")
    val criterion = new ClassNLLCriterion[Float]()
    val (model, size) = netType match {
      case "alexnet" => (com.intel.analytics.sparkdl.models.AlexNet[Float](classNum), 227)
      case "googlenet_v1" => (com.intel.analytics.sparkdl.models.GoogleNet_v1[Float](classNum), 224)
      case "googlenet_v2" => (com.intel.analytics.sparkdl.models.GoogleNet_v2[Float](classNum), 224)
    }
    println(model)
    val parameters = model.getParameters()._1
    val metrics = new Metrics

    val optM = getOptimMethodFloat(params.masterOptM)
    val dataSets = new ShuffleBatchDataSet[Int, Float](
      trainData, (d, t1, t2) => (t1.resize(Array(params.workerConfig[Int]("batch"), 3, size, size)),
        t2.resize(Array(params.workerConfig[Int]("batch"))).fill(1)),
      params.workerConfig[Int]("batch"), params.workerConfig[Int]("batch"))

    val pm = if (params.pmType == "allreduce") {
      new AllReduceParameterManager[Float](parameters, dataSets.partitions(), metrics)
    } else if (params.pmType == "onereduce") {
      new OneReduceParameterManager[Float](parameters, dataSets.partitions(), metrics)
    } else {
      throw new IllegalArgumentException()
    }

    val optimizer = new GradAggEpochOptimizer[Float](model, criterion, optM,
      pm, dataSets, metrics, params.masterConfig)
    optimizer.setMaxEpoch(100)

    optimizer.optimize()
  }
}
