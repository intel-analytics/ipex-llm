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

package com.intel.analytics.bigdl.models.lm

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text.{TensorSeqToBatch, TextSeqToTensorSeq}
import com.intel.analytics.bigdl.nn.CrossEntropyCriterion
import com.intel.analytics.bigdl.optim._
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.Logger


object Train {

  import Utils._
  val logger = Logger.getLogger(getClass)

  /**
    * This example demonstrates the use of fasttext for text classification.
    *
    * Based on Joulin et al's paper:
    *
    * Bags of Tricks for Efficient Text Classification
    * https://arxiv.org/abs/1607.01759
    *
    * Results on IMDB datasets with uni and bi-gram embeddings:
    *   Uni-gram:
    *   Bi-gram:
    */
  def main(args: Array[String]): Unit = {
    trainParser.parse(args, new TrainParams()).map(param => {
      var _maxFeatures = param.maxFeatures
      println("loading data...")
      val data = loadData(param.folder, Some(_maxFeatures))
      var _train: List[List[Int]] = data._1
      var _test: List[List[Int]] = data._3
      println(s"${_train.length} train sequence")
      println(s"${_test.length} test sequence")
      println(s"Average train sequence length: ${_train.map(_.length).sum / _train.length}")
      println(s"Average test sequence length: ${_test.map(_.length).sum / _test.length}")

      val ngramSet = scala.collection.mutable.Set[List[Int]]()
      if (param.ngramRange > 1) {
        println(s"Adding ${param.ngramRange}-gram features")
        // Create set of unique n-gram from the training set
        for (sample <- _train) {
          for (i <- 2 to param.ngramRange) {
            createNgramSet(sample, param.ngramRange).foreach(ngramSet.add)
          }
        }

        // Dictionary mapping n-gram token to a unique integer.
        // Integer values are greater than max_features in order
        // to avoid collision with existing features.
        val startIndex = _maxFeatures + 1
        val token2Index: List[(List[Int], Int)]
          = ngramSet.toList.zipWithIndex.map { case (token, i) => (token, (i + startIndex)) }

        // max_features is the highest integer that could be found in the dataset
        _maxFeatures = token2Index.map(t => t._2).max + 1

        // Augmenting _train and _test with n-grams features
        _train = addNgram(_train, token2Index.toMap, param.ngramRange)
        _test = addNgram(_test, token2Index.toMap, param.ngramRange)
      }

      println("padding sequences")
      val train: List[List[Int]] = padSequence(_train, maxLen = Some(param.maxLen))
      val test: List[List[Int]] = padSequence(_test, maxLen = Some(param.maxLen))
      val labelsTrain: List[List[Int]] = List(data._2)
      val labelsTest: List[List[Int]] = List(data._4)

      val trainData: Array[(Seq[Int], Seq[Int])] = (train zip labelsTrain).toArray
      val testData: Array[(Seq[Int], Seq[Int])] = (test zip labelsTest).toArray
      println(s"Average train sequence length: ${train.map(_.length).sum / train.length}")
      println(s"Average test sequence length: ${test.map(_.length).sum / test.length}")

      val trainSet = DataSet.array(trainData)
        .transform(TextSeqToTensorSeq(param.maxFeatures))
        .transform(TensorSeqToBatch())

      val validationSet = DataSet.array(testData)
        .transform(TextSeqToTensorSeq(param.maxFeatures))
        .transform(TensorSeqToBatch())

      println("Building model...")
      val model = FastText(
        inputSize = param.maxFeatures,
        hiddenSize = param.hiddenSize,
        outputSize = 1)
      model.reset()

      Engine.setCoreNumber(1)
      Engine.model.setPoolSize(param.coreNumber)

      val optimizer = Optimizer(
        model = model,
        dataset = trainSet,
        criterion = new CrossEntropyCriterion[Float]()
      )

      val state = T("learningRate" -> 0.002, "learningRateDecay" -> 0.0002)
      optimizer
        .setState(state)
        .setOptimMethod(new Adagrad[Float]())
        .setValidation(Trigger.everyEpoch, validationSet, Array(new Top1Accuracy[Float]))
        .setEndWhen(Trigger.maxEpoch(param.maxEpoch))
        .optimize()
    })
  }
}
