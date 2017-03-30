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
package com.intel.analytics.bigdl.models.utils

import com.intel.analytics.bigdl.models.lenet.LeNet5
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class ModelBroadcastSpec extends FlatSpec with Matchers with BeforeAndAfter {

  var sc: SparkContext = null

  Logger.getLogger("org").setLevel(Level.WARN)
  Logger.getLogger("akka").setLevel(Level.WARN)
  before {
    sc = new SparkContext(new SparkConf().setMaster("local[1]").setAppName("ModelBroadcast"))
  }

  "model broadcast" should "work properly" in {
    val model = LeNet5(10)

    val modelBroadCast = ModelBroadcast[Float].broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  "model broadcast with getParameters" should "work properly" in {
    val model = LeNet5(10)
    model.getParameters()

    val modelBroadCast = ModelBroadcast[Float].broadcast(sc, model)
    modelBroadCast.value().toString should be(model.toString)
    modelBroadCast.value().parameters()._1 should be(model.parameters()._1)
  }

  after {
    if (sc != null) {
      sc.stop()
    }
  }

}
