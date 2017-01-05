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
package com.intel.analytics.bigdl.models.embedding

import scopt.OptionParser

object Utils {
  /**
   * Options used by our word2vec model.
   *
   * @param saveLocation
   * @param trainDataLocation
   * @param testDataLocation
   * @param numNegSamples Number of negative samples per example.
   * @param embeddingSize Embedding dimension.
   * @param windowSize The number of words to predict to the left
   *                   and right of the target word.
   * @param minCount The minimum number of word occurrences for it
   *                 to be included in the vocabulary.
   * @param subsample Sub-sampling threshold for word occurrence.
   * @param alpha Negative sampling unigram distribution raised to alpha power
   * @param maxSentenceLength The maximum sentence length
   * @param learningRate
   * @param batchSize
   */
  case class Word2VecConfig(
    saveLocation: String = "",
    trainDataLocation: String = "",
    testDataLocation: String = "",
    numNegSamples: Int = 0,
    embeddingSize: Int = 200,
    windowSize: Int = 5,
    minCount: Int = 5,
    subsample: Double = 1e-3,
    alpha: Double = 0.75,
    maxSentenceLength: Int = 1000,
    learningRate: Double = 1e-3,
    batchSize: Int = 16,
    coreNumber: Int = -1,
    nodeNumber: Int = -1,
    maxEpoch: Int = 90,
    env: String = "local"
  )

  def parse(args: Array[String]): Word2VecConfig = new OptionParser[Word2VecConfig]("word2vec") {
    help("help") text "prints this usage text"
    opt[String]("saveLocation") required() action { (x, c) => c.copy(saveLocation = x) }
    opt[String]("trainDataLocation") required() action { (x, c) => c.copy(trainDataLocation = x) }
    opt[String]("testDataLocation")required() action { (x, c) => c.copy(testDataLocation = x) }
    opt[Int]("numNegSamples")
      .text("Negative samples per training example.")
      .action { (x, c) => c.copy(numNegSamples = x) }
    opt[Int]("embeddingSize")
      .text("Initial learning rate.")
      .action { (x, c) => c.copy(embeddingSize = x) }
    opt[Int]("windowSize")
      .text("The number of words to predict to the left and right of the target word.")
      .action { (x, c) => c.copy(windowSize = x) }
    opt[Int]("minCount")
      .text("The minimum number of word occurrences for it to be included in the vocabulary.")
      .action { (x, c) => c.copy(minCount = x) }
    opt[Double]("subsample")
      .text("Subsample threshold for word occurrence. Words that appear with higher " +
        "frequency will be randomly down-sampled. Set to 0 to disable.")
      .action { (x, c) => c.copy(subsample = x) }
    opt[Double]("maxSentenceLength")
      .text("The maximum threshold of sentence length accepted, if length than the threshold" +
        "the rest will be dropped")
      .action { (x, c) => c.copy(subsample = x) }
  }.parse(args, Word2VecConfig()).get
}
