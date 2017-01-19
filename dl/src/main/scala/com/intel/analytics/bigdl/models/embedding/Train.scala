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
package com.intel.analytics.bigdl.models.embedding

import com.intel.analytics.bigdl._
import com.intel.analytics.bigdl.dataset.{DataSet, MiniBatch, Sample, SampleToBatch}
import com.intel.analytics.bigdl.dataset.text.{LowerCase, Tokenizer}
import com.intel.analytics.bigdl.nn._
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, SGD, Trigger}
import com.intel.analytics.bigdl.utils.{Engine, File, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

import scala.collection.Parallel

object Train {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    Logger.getLogger("akka").setLevel(Level.ERROR)
    Logger.getLogger("breeze").setLevel(Level.ERROR)
    Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

    val params = Utils.parse(args)

    val sc = Engine.init(params.nodeNumber, params.coreNumber, params.env == "spark")
      .map(conf => {
        conf.setAppName("BigDL Word2Vec Example")
          .set("spark.task.maxFailures", "1")
          .setMaster("local[4]")
        new SparkContext(conf)
      }).get

    val word2Vec = Word2Vec(params)

    val tokens =
      (DataSet.rdd(sc.textFile(params.trainDataLocation))
      -> LowerCase()
      -> Tokenizer())

    val rddTokens = tokens.toDistributed().data(false)

    word2Vec.initialize(rddTokens)

    val trainSetRDD = rddTokens
      .mapPartitions[Sample[Float]](word2Vec.generateTrainingData().apply)

    val trainSet = DataSet.rdd(trainSetRDD) -> SampleToBatch(params.batchSize)

    val model = if (params.modelSnapshot.isDefined) {
      Module.load[Float](params.modelSnapshot.get)
    } else {
      word2Vec.getModel
    }

//    word2Vec.wordVectors.weight.apply1(x => 1)
    val optimizer = Optimizer(
      model = model,
//      dataset = a,
      dataset = trainSet,
      criterion = BCECriterion[Float]())

    if (params.checkpoint.isDefined) {
      optimizer.setCheckpoint(params.checkpoint.get, Trigger.everyEpoch)
    }

    val state = if (params.stateSnapshot.isDefined) {
      T.load(params.stateSnapshot.get)
    } else {
      T(
        "learningRate" -> params.learningRate
      )
    }

    val trained = optimizer
      .setState(state)
      .setOptimMethod(new SGD())
      .setEndWhen(Trigger.maxEpoch(params.maxEpoch))
      .optimize()

    word2Vec.wordVectors = trained.asInstanceOf[Sequential[Float]]
      .modules(1).asInstanceOf[ParallelTable[Float]].modules(1).asInstanceOf[LookupTable[Float]]

    word2Vec.normalizeWordVectors()

    word2Vec.printSimilarWords(
      Array("usa", "like", "play", "go", "china", "the", "he", "can"), 5)
  }
}
