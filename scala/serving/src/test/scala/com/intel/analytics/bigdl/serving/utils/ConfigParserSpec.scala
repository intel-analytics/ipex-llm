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

package com.intel.analytics.bigdl.serving.utils

import org.scalatest.{FlatSpec, Matchers}
import com.intel.analytics.bigdl.serving.utils.AssertUtils

class ConfigParserSpec extends FlatSpec with Matchers {
  val configPath = getClass.getClassLoader.getResource("serving").getPath + "/config-test.yaml"

  val configParser = new ConfigParser(configPath)
  "load set config" should "work" in {
    val conf = configParser.loadConfig()
    AssertUtils.conditionFailTest(conf.modelPath.isInstanceOf[String])
    AssertUtils.conditionFailTest(conf.modelPath == "/path")
    AssertUtils.conditionFailTest(conf.modelParallelism == 10)
    AssertUtils.conditionFailTest(conf.inputAlreadyBatched.isInstanceOf[Boolean])
    AssertUtils.conditionFailTest(conf.inputAlreadyBatched == true)
    AssertUtils.conditionFailTest(conf.redisSecureTrustStorePassword.isInstanceOf[String])
  }
  "load default config" should "work" in {
    val conf = configParser.loadConfig()
    AssertUtils.conditionFailTest(conf.threadPerModel == 1)
    AssertUtils.conditionFailTest(conf.postProcessing == "")
  }
}
