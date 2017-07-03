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
package com.intel.analytics.bigdl.example.treeLSTMSentiment

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.example.treeLSTMSentiment.Utils._
import com.intel.analytics.bigdl.nn.{TimeDistributedCriterion, _}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.utils.{Engine, LoggerFilter, T}
import org.apache.log4j.{Level => Levle4j, Logger => Logger4j}
import org.apache.spark.SparkContext
import org.slf4j.{Logger, LoggerFactory}

import scala.language.existentials

object Train {
  val log: Logger = LoggerFactory.getLogger(this.getClass)
  LoggerFilter.redirectSparkInfoLogs()
  Logger4j.getLogger("com.intel.analytics.bigdl.optim").setLevel(Levle4j.INFO)

  def main(args: Array[String]): Unit = {
    val params = paramParser.parse(args, TreeLSTMSentimentParam()).get
    train(params)
  }

  def train(param: TreeLSTMSentimentParam): Unit = {
    val DATA_DIR = param.baseDir
    val classNum = if (param.fineGrained) 5 else 3
    val criterion = TimeDistributedCriterion(ClassNLLCriterion())
    val conf = Engine.createSparkConf()
      .setAppName("Text classification")
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)
    Engine.init

    val paddingValue = 1
    val oovChar = 2
    val indexFrom = 3
    val labelPadding = if (param.fineGrained) 4f else 3f
    val glovePath = s"$DATA_DIR/glove/glove.840B.300d.txt"
    val vocabPath = s"$DATA_DIR/sst/vocab-cased.txt"
    val (word2VecTensor, vocab) =
      loadEmbeddingAndVocabulary(glovePath, vocabPath, indexFrom)

    val vocabBC = sc.broadcast(vocab)
    val (trainTreeRDD, trainLabelRDD, trainSentenceRDD) = preProcessData(
      sc,
      vocabBC,
      param.fineGrained,
      oovChar,
      s"$DATA_DIR/sst/train/parents.txt",
      s"$DATA_DIR/sst/train/labels.txt",
      s"$DATA_DIR/sst/train/sents.txt")
    println(
      s"""
         |train treeRDD count: ${trainTreeRDD.count()}
         |train labelRDD count: ${trainLabelRDD.count()}
         |train sentenceRDD count: ${trainSentenceRDD.count()}
      """.stripMargin)

    val (devTreeRDD, devLabelRDD, devSentenceRDD) = preProcessData(
      sc,
      vocabBC,
      param.fineGrained,
      oovChar,
      s"$DATA_DIR/sst/dev/parents.txt",
      s"$DATA_DIR/sst/dev/labels.txt",
      s"$DATA_DIR/sst/dev/sents.txt")
    println(
      s"""
         |dev treeRDD count: ${devTreeRDD.count()}
         |dev labelRDD count: ${devLabelRDD.count()}
         |dev sentenceRDD count: ${devSentenceRDD.count()}
      """.stripMargin)

    val trainRDD = toSample(trainTreeRDD, trainLabelRDD, trainSentenceRDD, param.fineGrained)
    val devRDD = toSample(devTreeRDD, devLabelRDD, devSentenceRDD, param.fineGrained)

    val optimizer = Optimizer(
      model = TreeLSTMSentiment(word2VecTensor, param.hiddenSize, classNum),
      sampleRDD = trainRDD,
      criterion = criterion,
      batchSize = param.batchSize,
      isInOrder = false,
      featurePaddings = Some(Array(Tensor(T(paddingValue.toFloat)), Tensor(T(-1f, -1f, -1f)))),
      labelPadding = Some(labelPadding)
    )
    optimizer
      .setOptimMethod(new Adagrad(
        learningRate = param.learningRate,
        weightDecay = param.regRate))
      .setValidation(
        Trigger.everyEpoch,
        devRDD,
        Array(new TreeNNAccuracy(param.fineGrained)),
        param.batchSize,
        featurePaddings =
          Some(Array(Tensor(T(paddingValue.toFloat)), Tensor(T(-1f, -1f, -1f)))),
        labelPadding = Some(labelPadding))
      .setEndWhen(Trigger.maxEpoch(20))
      .optimize()
    sc.stop()
  }
}
