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

import java.nio.ByteBuffer
import java.nio.file.{Files, Path}

import com.intel.analytics.bigdl.dataset.Sample
import scopt.OptionParser

object Utils {
  val trainMean = 0.13066047740239506
  val trainStd = 0.3081078

  val testMean = 0.13251460696903547
  val testStd = 0.31048024

  case class TrainParams(
    folder: String = "./",
    checkpoint: Option[String] = None,
    modelSnapshot: Option[String] = None,
    stateSnapshot: Option[String] = None,
    batchSize: Int = 10,
    learningRate: Double = 0.05,
    maxEpoch: Int = 15,
    coreNumber: Int = -1,
    nodeNumber: Int = -1,
    env: String = "local"
  )

  val trainParser = new OptionParser[TrainParams]("BigDL Lenet Train Example") {
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

    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))

    opt[Double]('r', "learningRate")
      .text("learning rate")
      .action((x, c) => c.copy(learningRate = x))

    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))

    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumber = x))
      .required()
    opt[Int]('n', "node")
      .text("node number to train the model")
      .action((x, c) => c.copy(nodeNumber = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]("env")
      .text("execution environment")
      .validate(x => {
        if (Set("local", "spark").contains(x.toLowerCase)) {
          success
        } else {
          failure("env only support local|spark")
        }
      })
      .action((x, c) => c.copy(env = x.toLowerCase()))
      .required()
  }

  case class TestParams(
    folder: String = "./",
    model: String = "",
    coreNumber: Int = -1,
    nodeNumber: Int = -1,
    batchSize: Int = 128,
    env: String = "local"
  )

  val testParser = new OptionParser[TestParams]("BigDL Lenet Test Example") {
    opt[String]('f', "folder")
      .text("where you put the MNIST data")
      .action((x, c) => c.copy(folder = x))

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
      .required()
    opt[Int]('c', "core")
      .text("cores number on each node")
      .action((x, c) => c.copy(coreNumber = x))
      .required()
    opt[Int]('n', "nodeNumber")
      .text("nodes number to train the model")
      .action((x, c) => c.copy(nodeNumber = x))
      .required()
    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))
    opt[String]("env")
      .text("execution environment")
      .validate(x => {
        if (Set("local", "spark").contains(x.toLowerCase)) {
          success
        } else {
          failure("env only support local|spark")
        }
      })
      .action((x, c) => c.copy(env = x.toLowerCase()))
      .required()
  }

  private[bigdl] def load(featureFile: Path, labelFile: Path): Array[Sample] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(labelFile))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(featureFile))
    val labelMagicNumber = labelBuffer.getInt()

    require(labelMagicNumber == 2049)
    val featureMagicNumber = featureBuffer.getInt()
    require(featureMagicNumber == 2051)

    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)

    val rowNum = featureBuffer.getInt()
    val colNum = featureBuffer.getInt()

    val result = new Array[Sample](featureCount)
    var i = 0
    while (i < featureCount) {
      val img = new Array[Byte]((rowNum * colNum))
      var y = 0
      while (y < rowNum) {
        var x = 0
        while (x < colNum) {
          img(x + y * colNum) = featureBuffer.get()
          x += 1
        }
        y += 1
      }
      result(i) = Sample(img, labelBuffer.get().toFloat + 1.0f)
      i += 1
    }

    result
  }
}
