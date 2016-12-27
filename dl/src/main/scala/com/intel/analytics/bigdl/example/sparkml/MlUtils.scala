/*
 * Licensed to Intel Corporation under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * Intel Corporation licenses this file to You under the Apache License, Version 2.0
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
package com.intel.analytics.bigdl.example.sparkml

import scopt.OptionParser

object MlUtils {

  val testMean = 0.13251460696903547
  val testStd = 0.31048024

  case class PredictParams(
    folder: String = "./",
    model: String = "",
    partitionNum : Int = 4,
    coreNumber: Int = -1,
    nodeNumber: Int = -1
  )

  val predictParser = new OptionParser[PredictParams]("BigDL Predict Example") {
    opt[String]('f', "folder")
      .text("where you put the test data")
      .action((x, c) => c.copy(folder = x))
      .required()

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
      .required()

    opt[Int]("partitionNum")
      .text("partition num")
      .action((x, c) => c.copy(partitionNum = x))
      .required()

    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumber = x))

    opt[Int]('n', "nodeNumber")
      .text("nodes number to train the model")
      .action((x, c) => c.copy(nodeNumber = x))
  }
}
