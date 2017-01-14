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

package com.intel.analytics.bigdl.example.finetune_flickr_style

import scopt.OptionParser

object Options {
  case class TrainParams(
    folder: String = "./",
    modelName: String = "GoogleNet",
    checkpoint: Option[String] = None,
    caffeDefPath: String = "",
    modelPath: String = "",
    batchSize: Int = 32,
    learningRate: Double = 0.001,
    maxEpoch: Int = 60,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
//    coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2),
    coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2),
    nodesNumber: Int = 1,
    env: String = "local"
  )

  val trainParser = new OptionParser[TrainParams]("BigDL FineTune Example") {
    head("Train Flickr Style model on single node")
    opt[String]('f', "folder")
      .text("where you put your local data/seq files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
    opt[String]("modelName")
      .text("model name")
      .action((x, c) => c.copy(modelName = x))
    opt[String]("caffeDefPath")
      .text("caffe model definition file")
      .action((x, c) => c.copy(caffeDefPath = x))
    opt[String]("modelPath")
      .text("existing model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[Double]('l', "learningRate")
      .text("Learning Rate")
      .action((x, c) => c.copy(learningRate = x))
    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))
    opt[String]("env")
      .text("running environment, should be local or spark")
      .action((x, c) => c.copy(env = x))
    opt[Int]('n', "node")
      .text("node number to train the model")
      .action((x, c) => c.copy(nodesNumber = x))
  }

  case class TestParams(
                         folder: String = "./",
                         model: String = "",
                         coreNumber: Int = -1,
                         nodeNumber: Int = -1,
                         batchSize: Int = 50,
                         env: String = "local"
                        )

  val testParser = new OptionParser[TestParams]("BigDL FineTune Example") {
    head("Train Flickr Style model on single node")
    opt[String]('f', "folder")
      .text("where you put your local data/seq files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]("env")
      .text("running environment, should be local or spark")
      .action((x, c) => c.copy(env = x))
    opt[Int]('n', "nodes")
      .text("node number to train the model")
      .action((x, c) => c.copy(nodeNumber = x))
  }
}
