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

package com.intel.analytics.bigdl.models.lenet

import scopt.OptionParser

object Options {
  case class TrainLocalParams(
    folder: String = "./",
    cache: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    batchSize: Int = 32,
    learningRate: Double = 0.05,
    maxEpoch: Int = 10,
    coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2)
  )

  val trainLocalParser = new OptionParser[TrainLocalParams]("BigDL Lenet Example") {
    head("Train Lenet model on single node")
    opt[String]('f', "folder")
      .text("where you put the MNIST data")
      .action((x, c) => c.copy(folder = x))

    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))

    opt[String]("cache")
      .text("where to cache the model")
      .action((x, c) => c.copy(cache = Some(x)))

    opt[Double]('r', "learningRate")
      .text("learning rate")
      .action((x, c) => c.copy(learningRate = x))

    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))

    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
  }

  case class TrainSparkParams(
    folder: String = "./",
    cache: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    batchSize: Int = 32,
    learningRate: Double = 0.05,
    maxEpoch: Int = 10,
    coreNumberPerNode: Int = -1,
    nodesNumber: Int = -1
  )

  val trainSparkParser = new OptionParser[TrainSparkParams]("BigDL Lenet Example") {
    head("Train Lenet model on Apache Spark")
    opt[String]('f', "folder")
      .text("where you put the MNIST data")
      .action((x, c) => c.copy(folder = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("cache")
      .text("where to cache the model")
      .action((x, c) => c.copy(cache = Some(x)))
    opt[Double]('r', "learningRate")
      .text("learning rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumberPerNode = x))
      .required()
    opt[Int]('n', "nodeNumber")
      .text("nodes number to train the model")
      .action((x, c) => c.copy(nodesNumber = x))
      .required()
  }
}
