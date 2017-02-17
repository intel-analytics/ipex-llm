
package com.intel.analytics.bigdl.example.udf

import scopt.OptionParser
/**
  * Created by jwang on 2/16/17.
  */
object Options {

  /**
    * Parameters for text classification
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
                                      showNum: Int = 20)

  val localParser = new OptionParser[TextClassificationParams]("BigDL Example") {
    opt[String]('b', "baseDir")
      .required()
      .text("Base dir containing the training and word2Vec data")
      .action((x, c) => c.copy(baseDir = x))
    opt[String]('o', "coreNum")
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
  }

}
