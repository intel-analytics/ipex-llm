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

package com.intel.analytics.bigdl.models.vgg

import scopt.OptionParser

object Utils {
  val trainMean = (0.4913996898739353, 0.4821584196221302, 0.44653092422369434)
  val trainStd = (0.24703223517429462, 0.2434851308749409, 0.26158784442034005)
  val testMean = (0.4942142913295297, 0.4851314002725445, 0.45040910258647154)
  val testStd = (0.2466525177466614, 0.2428922662655766, 0.26159238066790275)

  case class TrainLocalParams(
    folder: String = "./",
    checkpointPath: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2)
  )

  val trainLocalParser = new OptionParser[TrainLocalParams]("BigDL Vgg on Cifar Example") {
    head("Train Vgg model on single node")
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpointPath = Some(x)))
    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
  }

  case class TrainSparkParams(
    folder: String = "./",
    checkpointPath: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    coreNumberPerNode: Int = -1,
    nodesNumber: Int = -1,
    batchSize: Option[Int] = None
  )

  val trainSparkParser = new OptionParser[TrainSparkParams]("BigDL Vgg on Cifar Example") {
    head("Train Vgg model on Apache Spark")
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpointPath = Some(x)))
    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumberPerNode = x))
      .required()
    opt[Int]('n', "nodeNumber")
      .text("nodes number to train the model")
      .action((x, c) => c.copy(nodesNumber = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = Some(x)))
  }

  case class TestLocalParams(
    folder: String = "./",
    model: String = "",
    coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2)
  )

  val testLocalParser = new OptionParser[TestLocalParams]("BigDL Vgg on Cifar10 Test Example") {
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
  }

  case class TestSparkParams(
    folder: String = "./",
    model: String = "",
    coreNumberPerNode: Int = -1,
    nodesNumber: Int = -1
  )

  val testSparkParser = new OptionParser[TestSparkParams]("BigDL Vgg on Cifar10 Test Example") {
    opt[String]('f', "folder")
      .text("where you put the Cifar10 data")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
      .required()
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
