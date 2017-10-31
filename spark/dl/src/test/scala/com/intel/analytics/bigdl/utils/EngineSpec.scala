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
    System.setProperty("bigdl.localMode", "true")
    Engine.init
    Engine.nodeNumber should be(1)
    System.clearProperty("bigdl.localMode")
  }

  it should "be inited correct under spark local environment" in {
    val conf = Engine.createSparkConf().setAppName("EngineSpecTest").setMaster("local[4]")
    sc = SparkContext.getOrCreate(conf)
    Engine.init
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(4)
  }

  it should "be inited correct localmode under no spark environment with java property" in {
    val property = "bigdl.localMode"
    System.setProperty(property, "true")
    Engine.init
    Engine.nodeNumber should be(1)
    System.clearProperty(property)
  }

  it should "be inited correct coreNumber under no spark environment with java property" in {
    val localMode = "bigdl.localMode"
    val coreNumber = "bigdl.coreNumber"
    System.setProperty("bigdl.localMode", "true")
    System.setProperty(coreNumber, "3")

    val tmp = System.getProperty("bigdl.localMode")

    Engine.init
    Engine.localMode should be(true)
    Engine.nodeNumber should be(1)
    Engine.coreNumber should be(3)

    System.clearProperty("bigdl.localMode")
    System.clearProperty(coreNumber)
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
      "spark.shuffle.reduceLocality.enabled" -> "false",
      "spark.shuffle.blockTransferService" -> "nio",
      "spark.scheduler.minRegisteredResourcesRatio" -> "1.0",
      "spark.speculation" -> "false"
    )
    conf.length should be(target.keys.size)
    conf.foreach(s => {
      s._2 should be(target(s._1))
    })
  }

  "readConf" should "skip blockTransferService if bigdl.network.nio is set to false" in {
    System.setProperty("bigdl.network.nio", "false")
    val conf = Engine.readConf
    val target = Map(
      "spark.shuffle.reduceLocality.enabled" -> "false",
      "spark.scheduler.minRegisteredResourcesRatio" -> "1.0",
      "spark.speculation" -> "false"
    )
    conf.length should be(target.keys.size)
    conf.foreach(s => {
      s._2 should be(target(s._1))
    })
    System.clearProperty("bigdl.network.nio")
  }

  "LocalMode" should "false if onSpark" in {
    intercept[IllegalArgumentException] {
      System.setProperty("bigdl.localMode", "true")
      Engine.init(1, 1, true)
      System.clearProperty("bigdl.localMode")
    }
  }

  "LocalMode" should "true if not onSpark" in {
    intercept[IllegalArgumentException] {
      System.setProperty("bigdl.localMode", "false")
      Engine.init(1, 1, false)
      System.clearProperty("bigdl.localMode")
    }
  }

}
