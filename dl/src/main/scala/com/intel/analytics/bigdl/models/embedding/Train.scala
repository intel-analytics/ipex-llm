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
import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text.Tokenizer
import com.intel.analytics.bigdl.nn.{BCECriterion, Module}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.optim.{Optimizer, SGD, Trigger}
import com.intel.analytics.bigdl.utils.{Engine, T}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext

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

    Engine.setCoreNumber(params.coreNumber)

    val word2Vec = Word2Vec(params)

    val tokens = DataSet.rdd(sc.textFile(params.trainDataLocation)) -> Tokenizer()

    word2Vec.fit(tokens.toDistributed().data(false))

    val trainSet = tokens -> word2Vec.transformToBatch

    val model = if (params.modelSnapshot.isDefined) {
      Module.load[Float](params.modelSnapshot.get)
    } else {
      word2Vec.getModel
    }

    val optimizer = Optimizer(
      model = model,
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
    optimizer
      .setState(state)
      .setOptimMethod(new SGD())
      .setEndWhen(Trigger.maxEpoch(params.maxEpoch))
      .optimize()

    word2Vec.normalizeWordVectors()
  }
}
