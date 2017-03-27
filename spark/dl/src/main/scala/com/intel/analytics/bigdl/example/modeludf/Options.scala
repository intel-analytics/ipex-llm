/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.intel.analytics.bigdl.example.modeludf

import scopt.OptionParser

object Options {

    /**
    * Parameters for text classification
    *
    * @param baseDir           The root directory which containing the training and embedding data
    * @param maxSequenceLength number of the tokens
    * @param maxWordsNum       maximum word to be included
    * @param trainingSplit     percentage of the training data
    * @param batchSize         size of the mini-batch
    * @param embeddingDim      size of the embedding vector
    * @param coreNum           same idea of spark core
    * @param nodeNum           size of the cluster
    */
  case class TextClassificationParams(baseDir: String = "./",
                                      maxSequenceLength: Int = 1000,
                                      maxWordsNum: Int = 20000,
                                      trainingSplit: Double = 0.8,
                                      batchSize: Int = 128,
                                      embeddingDim: Int = 100,
                                      coreNum: Int = 4,
                                      nodeNum: Int = 1,
                                      partitionNum: Int = 4,
                                      modelPath: Option[String] = None,
                                      checkpoint: Option[String] = None,
                                      showNum: Int = 20,
                                      testDir: String = "./",
                                      bootstrapServer: String = "",
                                      topic: String = "",
                                      interval: Long = 5
                                     )

  val localParser = new OptionParser[TextClassificationParams]("BigDL Example") {
    opt[String]('b', "baseDir")
      .text("Base dir containing the training and word2Vec data")
      .action((x, c) => c.copy(baseDir = x))
    opt[String]('c', "coreNum")
      .text("core number")
      .action((x, c) => c.copy(coreNum = x.toInt))
    opt[String]('n', "nodeNum")
      .text("nodeNumber")
      .action((x, c) => c.copy(nodeNum = x.toInt))
    opt[String]('p', "partitionNum")
      .text("you may want to tune the partitionNum if run into spark mode")
      .action((x, c) => c.copy(partitionNum = x.toInt))
    opt[String]('s', "maxSequenceLength")
      .text("maxSequenceLength")
      .action((x, c) => c.copy(maxSequenceLength = x.toInt))
    opt[String]('w', "maxWordsNum")
      .text("maxWordsNum")
      .action((x, c) => c.copy(maxWordsNum = x.toInt))
    opt[String]('l', "trainingSplit")
      .text("trainingSplit")
      .action((x, c) => c.copy(trainingSplit = x.toDouble))
    opt[String]('z', "batchSize")
      .text("batchSize")
      .action((x, c) => c.copy(batchSize = x.toInt))
    opt[String]("modelPath")
      .text("where to load the model")
      .action((x, c) => c.copy(modelPath = Some(x)))
    opt[String]("checkpoint")
      .text("where to load the model")
      .action((x, c) => c.copy(checkpoint = Some(x)))
    opt[String]("showNum")
      .text("how many rows been shown in sql result")
      .action((x, c) => c.copy(showNum = x.toInt))
    opt[String]("bootstrap")
      .text("Boot strap server")
      .action((x, c) => c.copy(bootstrapServer = x))
    opt[String]("topic")
      .text("Subscribed Topic")
      .action((x, c) => c.copy(topic = x))
    opt[Long]('i', "interval")
      .text("consume interval")
      .action((x, c) => c.copy(interval = x))
    opt[String]('f', "dataDir")
      .text("Text dir containing the text data")
      .action((x, c) => c.copy(testDir = x))
  }

    /**
    * Parameters for text producer
    *
    * @param folder      directory of text data
    * @param brokerList  broker list
    * @param targetTopic publish topic
    * @param interval    publish interval
    */
  case class TextKafkaProducerParams(
                                      folder: String = "./",
                                      brokerList: String = "",
                                      targetTopic: String = "",
                                      interval: Long = 1,
                                      batchSize: Int = 2)

  val kafkaProducerParser = new OptionParser[TextKafkaProducerParams]("BigDL Streaming Example") {
    opt[String]('f', "folder")
      .required()
      .text("Base dir containing the text data")
      .action((x, c) => c.copy(folder = x))
    opt[String]('b', "brokerList")
      .text("broker list")
      .action((x, c) => c.copy(brokerList = x))
    opt[String]('t', "topic")
      .text("topic")
      .action((x, c) => c.copy(targetTopic = x))
    opt[Long]('i', "interval")
      .text("produce interval")
      .action((x, c) => c.copy(interval = x))
    opt[Int]("batchsize")
      .text("produce batchsize")
      .action((x, c) => c.copy(batchSize = x))
  }

      /**
    * Text parquet producer parameters
    *
    * @param srcFolder
    * @param destFolder
    * @param interval
    */
  case class TextProducerParquetParams(
                                        srcFolder: String = "./",
                                        destFolder: String = "./",
                                        batchsize: Int = 2,
                                        interval: Long = 5)

  val parquetProducerParser
  = new OptionParser[TextProducerParquetParams]("BigDL Streaming Example") {
    opt[String]('s', "srcFolder")
      .required()
      .text("Base dir containing the text data")
      .action((x, c) => c.copy(srcFolder = x))
    opt[String]('d', "destFolder")
      .required()
      .text("Destination parquet dir containing the text data")
      .action((x, c) => c.copy(destFolder = x))
    opt[Int]('b', "batchsize")
      .text("produce batchsize")
      .action((x, c) => c.copy(batchsize = x))
    opt[Long]('i', "interval")
      .text("produce interval")
      .action((x, c) => c.copy(interval = x))
  }
}
