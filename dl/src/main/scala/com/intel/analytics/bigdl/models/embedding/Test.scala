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

import com.intel.analytics.bigdl.dataset.DataSet
import com.intel.analytics.bigdl.dataset.text.{LowerCase, Tokenizer}
import com.intel.analytics.bigdl.utils.{Engine, File}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.SparkContext
import org.slf4j.{LoggerFactory, Logger => sl4jLogger}

object Test {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.bigdl.optim").setLevel(Level.INFO)

  val log: sl4jLogger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    val params = Utils.parse(args)

    val sc = Engine.init(params.nodeNumber, params.coreNumber, onSpark = true)
      .map(conf => {
        conf.setAppName("BigDL Word2Vec Example")
          .set("spark.task.maxFailures", "1")
          .setMaster("local[4]")
        new SparkContext(conf)
      }).get

    val testSet = (DataSet.rdd(sc.textFile(params.testDataLocation))
      -> Tokenizer())

    var model: Word2Vec = null
    if (params.modelLocation == null && params.modelLocation.equals("")) {
      log.error("Model directory should be set")
      System.exit(0)
    } else {
      model = File.load[Word2Vec](params.modelLocation + "/word2vec.obj")
    }

    Engine.setCoreNumber(params.coreNumber)

    testSet
      .toDistributed()
      .data(false)
      .foreach(words => model.printWordAnalogy(words.toArray, params.numClosestWords))
  }
}
