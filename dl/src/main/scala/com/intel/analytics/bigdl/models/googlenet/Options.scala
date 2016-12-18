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

package com.intel.analytics.bigdl.models.googlenet

import scopt.OptionParser

object Options {

  case class TrainParams(
    folder: String = "./",
    cache: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    coreNumberPerNode: Int = -1,
    nodesNumber: Int = -1,
    batchSize: Option[Int] = None
  )

  val trainParser = new OptionParser[TrainParams]("BigDL Lenet Example") {
    head("Train GoogleNet model on Apache Spark")
    opt[String]('f', "folder")
      .text("url of hdfs folder store the hadoop sequence files")
      .action((x, c) => c.copy(folder = x))
    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))
    opt[String]("cache")
      .text("where to cache the model")
      .action((x, c) => c.copy(cache = Some(x)))
    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumberPerNode = x))
      .required()
    opt[Int]('n', "nodeNumber")
      .text("nodes number to train the model")
      .action((x, c) => c.copy(nodesNumber = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batchi size")
      .action((x, c) => c.copy(batchSize = Some(x)))
  }

  case class TestParams(
    folder: String = "./",
    model: String = "",
    coreNumberPerNode: Int = -1,
    nodesNumber: Int = -1
  )

  val testParser = new OptionParser[TestParams]("BigDL Googlenet Test Example") {
    opt[String]('f', "folder")
      .text("url of hdfs folder store the hadoop sequence files")
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
