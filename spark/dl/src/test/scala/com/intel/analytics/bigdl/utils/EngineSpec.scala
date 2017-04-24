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

package com.intel.analytics.bigdl.utils

import org.apache.spark.SparkContext
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class EngineSpec extends FlatSpec with Matchers with BeforeAndAfter {
  private var sc: SparkContext = _

  before {
    Engine.reset()
  }

  after {
    if (sc != null) {
      sc.stop()
      sc = null
    }
    Engine.reset()
  }

  "Engine" should "initialize correctly under Spark local environment" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("local[4]")
    val sc = SparkContext.getOrCreate(conf)
    Engine.init(conf)
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(4)
  }

  "Engine" should "initialize with correct value under Spark local environment" in {
    Engine.init(1, 4, true)
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(4)
  }

  "Engine" should "initialize correctly under Spark standalone environment" in {
      val conf = Engine.createSparkConf().setAppName("EngineSpecTest")
        .setMaster("spark://localhost:1234")
        .set("spark.executor.cores", "4").set("spark.cores.max", "24")
    val sc = SparkContext.getOrCreate(conf)
      Engine.init(conf)
      Engine.nodeNumber should be(6)
      Engine.coreNumber should be(4)
  }

  "Engine" should "initialize correctly under Spark standalone environment with 1 executor" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest")
      .setMaster("spark://localhost:1234")
      .set("spark.executor.cores", "4").set("spark.cores.max", "4")
    val sc = SparkContext.getOrCreate(conf)
    Engine.init(conf)
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(4)
  }

  "Engine" should "initialize correctly under YARN environment" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("local[*]")
      .set("spark.executor.cores", "4").set("spark.executor.instances", "6")
    val sc = SparkContext.getOrCreate(conf)
    conf.setMaster("yarn") // hack: can't actually init this master so only set here
    Engine.init(conf)
    Engine.nodeNumber should be(6)
    Engine.coreNumber should be(4)
  }

  "Engine" should "initialize correctly under YARN environment with 1 executor" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("local[*]")
      .set("spark.executor.cores", "4").set("spark.executor.instances", "1")
    val sc = SparkContext.getOrCreate(conf)
    conf.setMaster("yarn") // hack: can't actually init this master so only set here
    Engine.init(conf)
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(4)
  }

  "Engine" should "initialize correctly under Spark Mesos environment" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("local[*]")
      .set("spark.executor.cores", "4").set("spark.cores.max", "24")
    val sc = SparkContext.getOrCreate(conf)
    conf.setMaster("mesos://localhost:1234") // hack: can't actually init this master so set here
    Engine.init(conf)
    Engine.nodeNumber should be(6)
    Engine.coreNumber should be(4)
  }

  "Engine" should "initialize correctly under Spark Mesos environment with 1 executor" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("local[*]")
      .set("spark.executor.cores", "4").set("spark.cores.max", "4")
    val sc = SparkContext.getOrCreate(conf)
    conf.setMaster("mesos://localhost:1234") // hack: can't actually init this master so set here
    Engine.init(conf)
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(4)
  }

  "sparkExecutorAndCore" should "parse local[*]" in {
    val (nExecutor, _) =
      Engine.sparkExecutorAndCore(Engine.createSparkConf().setMaster("local[*]"), true).get
    nExecutor should be(1)
  }

  "readConf" should "be right" in {
    val conf = Engine.readConf
    val target = Map(
      "spark.executorEnv.DL_ENGINE_TYPE" -> "mklblas",
      "spark.executorEnv.MKL_DISABLE_FAST_MM" -> "1",
      "spark.executorEnv.KMP_BLOCKTIME" -> "0",
      "spark.executorEnv.OMP_WAIT_POLICY" -> "passive",
      "spark.executorEnv.OMP_NUM_THREADS" -> "1",
      "spark.yarn.appMasterEnv.DL_ENGINE_TYPE" -> "mklblas",
      "spark.yarn.appMasterEnv.MKL_DISABLE_FAST_MM" -> "1",
      "spark.yarn.appMasterEnv.KMP_BLOCKTIME" -> "0",
      "spark.yarn.appMasterEnv.OMP_WAIT_POLICY" -> "passive",
      "spark.yarn.appMasterEnv.OMP_NUM_THREADS" -> "1",
      "spark.shuffle.reduceLocality.enabled" -> "false",
      "spark.shuffle.blockTransferService" -> "nio",
      "spark.scheduler.minRegisteredResourcesRatio" -> "1.0"
    )
    conf.length should be(target.keys.size)
    conf.foreach(s => {
      s._2 should be(target(s._1))
    })
  }

  "SparkContext" should "be inited when call Engine.init" in {
    intercept[IllegalArgumentException] {
      Engine.init(Engine.createSparkConf())
    }
  }

}
