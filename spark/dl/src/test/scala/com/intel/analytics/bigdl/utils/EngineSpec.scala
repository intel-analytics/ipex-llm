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

import com.intel.analytics.bigdl.utils.{TestUtils => BigDLTestUtils}
import org.apache.spark.SparkEnvHelper
import org.scalatest.mock.MockitoSugar
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class EngineSpec extends FlatSpec with Matchers with BeforeAndAfter with MockitoSugar {

  after {
    Engine.reset
  }

  "Engine" should "be inited correct under no spark environment" in {
    Engine.localMode = true
    Engine.init
    Engine.nodeNumber should be(1)
  }

  "Engine" should "be inited correct under spark local environment" in {
    SparkEnvHelper.sparkLocalEnv(core = 4) {
      Engine.init
      Engine.nodeNumber should be(1)
      Engine.coreNumber should be(4)
    }
  }

  "Engine" should "be inited with correct value under spark local environment" in {
    SparkEnvHelper.sparkLocalEnv(core = 4) {
      Engine.init(1, 4, true)
      Engine.nodeNumber should be(1)
      Engine.coreNumber should be(4)
    }
  }

  "Engine" should "be inited correct under spark standalone environment" in {
    SparkEnvHelper.sparkStandaloneEnv(totalCore = 24, core = 4) {
      Engine.init
      Engine.nodeNumber should be(6)
      Engine.coreNumber should be(4)
    }
  }

  "Engine" should "be inited correct under spark standalone environment with single executor" in {
    SparkEnvHelper.sparkStandaloneEnv(totalCore = 4, core = 4) {
      Engine.init
      Engine.nodeNumber should be(1)
      Engine.coreNumber should be(4)
    }
  }

  "Engine" should "be inited correct under spark yarn environment" in {
    SparkEnvHelper.sparkYarnEnv(executors = 6, core = 4) {
      Engine.init
      Engine.nodeNumber should be(6)
      Engine.coreNumber should be(4)
    }
  }

  "Engine" should "be inited correct under spark yarn environment with single executor" in {
    SparkEnvHelper.sparkYarnEnv(executors = 1, core = 4) {
      Engine.init
      Engine.nodeNumber should be(1)
      Engine.coreNumber should be(4)
    }
  }

  "Engine" should "be inited correct under spark mesos environment" in {
    SparkEnvHelper.sparkMesosEnv(totalCore = 24, core = 4) {
      Engine.init
      Engine.nodeNumber should be(6)
      Engine.coreNumber should be(4)
    }
  }

  "Engine" should "be inited correct under spark mesos environment with single executor" in {
    SparkEnvHelper.sparkMesosEnv(totalCore = 4, core = 4) {
      Engine.init
      Engine.nodeNumber should be(1)
      Engine.coreNumber should be(4)
    }
  }

  "sparkExecutorAndCore" should "parse local[*]" in {
    val (nExecutor, executorCore) = Engine.sparkExecutorAndCore(true, "local[*]").get
    nExecutor should be(1)
    System.clearProperty("spark.master")
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
