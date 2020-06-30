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

package com.intel.analytics.bigdl.example.languagemodel

import scopt.OptionParser
import java.io._

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text._
import com.intel.analytics.bigdl.tensor.TensorNumericMath.TensorNumeric

import scala.util.Random
import org.apache.log4j.Logger
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.io.Source
import scala.reflect.ClassTag

object Utils {

  case class TrainParams(
                          dataFolder: String = "./",
                          modelSnapshot: Option[String] = None,
                          stateSnapshot: Option[String] = None,
                          checkpoint: Option[String] = None,
                          batchSize: Int = 20,
                          learningRate: Double = 0.01,
                          learningRateDecay: Double = 0.001,
                          hiddenSize: Int = 200,
                          vocabSize: Int = 10000,
                          nEpochs: Int = 4,
                          numLayers: Int = 2,
                          numSteps: Int = 20,
                          overWriteCheckpoint: Boolean = false,
                          keepProb: Float = 2.0f,
                          withTransformerModel: Boolean = false,
                          optimizerVersion: Option[String] = None)

  val trainParser = new OptionParser[TrainParams]("BigDL ptbModel Train Example") {
    opt[String]('f', "dataFolder")
      .text("where you put the text data")
      .action((x, c) => c.copy(dataFolder = x))
      .required()

    opt[String]("model")
      .text("model snapshot location")
      .action((x, c) => c.copy(modelSnapshot = Some(x)))
    opt[String]("state")
      .text("state snapshot location")
      .action((x, c) => c.copy(stateSnapshot = Some(x)))

    opt[String]("checkpoint")
      .text("where to cache the model and state")
      .action((x, c) => c.copy(checkpoint = Some(x)))

    opt[Int]('b', "batchSize")
      .text("batchSize of rnn")
      .action((x, c) => c.copy(batchSize = x))
      .required()

    opt[Double]('r', "learningRate")
      .text("learning rate")
      .action((x, c) => c.copy(learningRate = x))

    opt[Double]("learningRateDecay")
      .text("learningRateDecay")
      .action((x, c) => c.copy(learningRateDecay = x))

    opt[Int]('h', "hidden")
      .text("hidden size")
      .action((x, c) => c.copy(hiddenSize = x))

    opt[Int]("vocab")
      .text("dictionary length | vocabulary size")
      .action((x, c) => c.copy(vocabSize = x))

    opt[Int]('e', "nEpochs")
      .text("epoch numbers")
      .action((x, c) => c.copy(nEpochs = x))

    opt[Int]("numLayers")
      .text("number of recurrent layers")
      .action((x, c) => c.copy(numLayers = x))

    opt[Int]("numSteps")
      .text("number of words per record in LM")
      .action((x, c) => c.copy(numSteps = x))

    opt[Unit]("overWrite")
      .text("overwrite checkpoint files")
      .action( (_, c) => c.copy(overWriteCheckpoint = true) )

    opt[Double]("keepProb")
      .text("the probability p to do dropout")
      .action((x, c) => c.copy(keepProb = x.toFloat))

    opt[Boolean]("withTransformerModel")
      .text("Use transformer model in this LM")
      .action((x, c) => c.copy(withTransformerModel = true))

    opt[String]("optimizerVersion")
      .text("state optimizer version")
      .action((x, c) => c.copy(optimizerVersion = Some(x)))
  }
}
