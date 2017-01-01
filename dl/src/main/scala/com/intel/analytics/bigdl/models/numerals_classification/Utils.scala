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

package com.intel.analytics.bigdl.models.numerals_classification

import java.nio.ByteBuffer
import java.nio.file.{Files, Path}

import com.intel.analytics.bigdl.dataset.{MiniBatch, Transformer}
import com.intel.analytics.bigdl.tensor.{Storage, Tensor}
import scopt.OptionParser
import scala.collection.Iterator

object Utils {

  case class TrainParams(
                          folder: String = "./",
                          cache: Option[String] = None,
                          checkpoint: Option[String] = None,
                          modelSnapshot: Option[String] = None,
                          stateSnapshot: Option[String] = None,
                          batchSize: Int = 10,
                          learningRate: Double = 0.01,
                          maxEpoch: Int = 12,
                          coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2),
                          nodeNumber: Int = 1,
                          env: String = "local"
                        )

  val trainParser = new OptionParser[TrainParams]("BigDL Lenet Train Example") {
    opt[String]('f', "folder")
      .text("where you put the data")
      .action((x, c) => c.copy(folder = x))

    opt[Int]('b', "batchSize")
      .text("batch size")
      .action((x, c) => c.copy(batchSize = x))

    opt[String]("checkpoint")
      .text("where to cache the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))

    opt[String]("cache")
      .text("where to cache the model")
      .action((x, c) => c.copy(cache = Some(x)))

    opt[Int]('e', "maxEpoch")
      .text("epoch numbers")
      .action((x, c) => c.copy(maxEpoch = x))

    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))

    opt[Int]('n', "node")
      .text("nodes number to train the model")
      .action((x, c) => c.copy(nodeNumber = x))

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
                         coreNumber: Int = (Runtime.getRuntime().availableProcessors() / 2),
                         nodeNumber: Int = 1,
                         batchSize: Int = 10,
                         env: String = "local"
                       )

  val testParser = new OptionParser[TestParams]("BigDL classification Test Example") {
    opt[String]('f', "folder")
      .text("where you put the data")
      .action((x, c) => c.copy(folder = x))

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(model = x))
      .required()

    opt[Int]('c', "core")
      .text("cores number to train the model")
      .action((x, c) => c.copy(coreNumber = x))

    opt[Int]('n', "node")
      .text("nodes number to train the model")
      .action((x, c) => c.copy(nodeNumber = x))

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

  def load(featureFile: Path, labelFile: Path): Array[Sample] = {
    val labelBuffer = ByteBuffer.wrap(Files.readAllBytes(labelFile))
    val featureBuffer = ByteBuffer.wrap(Files.readAllBytes(featureFile))
    val labelCount = labelBuffer.getInt()
    val featureCount = featureBuffer.getInt()
    require(labelCount == featureCount)
    val arraySize = featureBuffer.getInt()

    val result = new Array[Sample](featureCount)
    var i = 0
    while (i < featureCount) {
      val features = new Array[Float]((arraySize))
      var x = 0
      while (x < arraySize) {
        features(x) = featureBuffer.getFloat()
        x += 1
      }
      result(i) = Sample(features, labelBuffer.getFloat() + 1)
      i += 1
    }
    result
  }

//  case class Sample(data: Array[Float], label: Float)

}

case class Sample(data: Array[Float], label: Float)

object SampleToBatch {
  def apply(batchSize : Int, featureSize: Int) : SampleToBatch = {
    new SampleToBatch(batchSize, featureSize)
  }
}

class SampleToBatch(batchSize: Int, featureSize: Int)
  extends Transformer[Sample, MiniBatch[Float]]{

  private def copySample(sample: Sample, storage: Array[Float], offset: Int): Unit = {
    val content = sample.data
    val frameLength = featureSize
    var j = 0
    while (j < frameLength) {
      storage(offset + j) = content(j)
      j += 1
    }
  }

  def apply(prev: Iterator[Sample]): Iterator[MiniBatch[Float]] = {
    new Iterator[MiniBatch[Float]] {
      private val featureTensor: Tensor[Float] = Tensor[Float]()
      private val labelTensor: Tensor[Float] = Tensor[Float]()
      private var featureData: Array[Float] = null
      private var labelData: Array[Float] = null

      override def hasNext: Boolean = prev.hasNext

      override def next(): MiniBatch[Float] = {
        if (prev.hasNext) {
          var i = 0
          while (i < batchSize && prev.hasNext) {
            val sample = prev.next()
            if (featureData == null) {
              featureData = new Array[Float](batchSize * featureSize)
              labelData = new Array[Float](batchSize)
            }
            copySample(sample, featureData, i * featureSize)
            labelData(i) = sample.label
            i += 1
          }
          if (labelTensor.nElement() != i) {
            featureTensor.set(Storage[Float](featureData),
              storageOffset = 1, sizes = Array(i, featureSize))
            labelTensor.set(Storage[Float](labelData),
              storageOffset = 1, sizes = Array(i))
          }
          MiniBatch(featureTensor, labelTensor)
        } else {
          null
        }
      }
    }
  }
}

