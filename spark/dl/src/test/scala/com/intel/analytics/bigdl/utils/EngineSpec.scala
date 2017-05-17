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
    sc = null
    Engine.reset
  }

  after {
    if (sc != null) {
      sc.stop()
      sc = null
    }
    Engine.reset
  }

  "Engine" should "be inited correct under no spark environment" in {
    Engine.localMode = true
    Engine.init
    Engine.nodeNumber should be(1)
  }

  it should "be inited correct under spark local environment" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("local[4]")
    sc = SparkContext.getOrCreate(conf)
    Engine.init
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(4)
  }

  it should "be inited with correct value under spark local environment" in {
    Engine.init(1, 4, true)
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(4)
  }

  it should "parse nodes, executors correctly for Spark standalone" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").
      setMaster("spark://localhost:1234").
      set("spark.cores.max", "24").set("spark.executor.cores", "4")
    Engine.parseExecutorAndCore(conf) should be(Some(6, 4))
  }

  it should "parse nodes, executors correctly for Spark standalone with single executor" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").
      setMaster("spark://localhost:1234").
      set("spark.cores.max", "4").set("spark.executor.cores", "4")
    Engine.parseExecutorAndCore(conf) should be(Some(1, 4))
  }

  it should "parse nodes, executors correctly for Spark YARN" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("yarn").
      set("spark.executor.instances", "6").set("spark.executor.cores", "4")
    Engine.parseExecutorAndCore(conf) should be(Some(6, 4))
  }

  it should "parse nodes, executors correctly for Spark YARN with single executor" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("yarn").
      set("spark.executor.instances", "1").set("spark.executor.cores", "4")
    Engine.parseExecutorAndCore(conf) should be(Some(1, 4))
  }

  it should "parse nodes, executors correctly for Spark Mesos" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").
      setMaster("mesos://localhost:1234").
      set("spark.cores.max", "24").set("spark.executor.cores", "4")
    Engine.parseExecutorAndCore(conf) should be(Some(6, 4))
  }

  it should "parse nodes, executors correctly for Spark Mesos with single executor" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").
      setMaster("mesos://localhost:1234").
      set("spark.cores.max", "4").set("spark.executor.cores", "4")
    Engine.parseExecutorAndCore(conf) should be(Some(1, 4))
  }

  "sparkExecutorAndCore" should "parse local[*]" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("local[*]")
    val (nExecutor, _) = Engine.parseExecutorAndCore(conf).get
    nExecutor should be(1)
  }

  "readConf" should "be right" in {
    val conf = Engine.readConf
    val target = Map(
      "spark.executorEnv.MKL_DISABLE_FAST_MM" -> "1",
      "spark.executorEnv.KMP_BLOCKTIME" -> "0",
      "spark.executorEnv.OMP_WAIT_POLICY" -> "passive",
      "spark.executorEnv.OMP_NUM_THREADS" -> "1",
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

  "LocalMode" should "false if onSpark" in {
    intercept[IllegalArgumentException] {
      Engine.localMode = true
      Engine.init(1, 1, true)
    }
  }

  "LocalMode" should "true if not onSpark" in {
    intercept[IllegalArgumentException] {
      Engine.localMode = false
      Engine.init(1, 1, false)
    }
  }

}
