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

package com.intel.analytics.bigdl.ppml.fl

import com.intel.analytics.bigdl.ppml.fl.example.DebugLogger
import com.intel.analytics.bigdl.ppml.fl.utils.PortUtils
import org.apache.logging.log4j.core.config.Configurator
import org.apache.logging.log4j.{Level, LogManager}
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FLSpec extends FlatSpec with Matchers with BeforeAndAfter {
  var port: Int = 8980
  var target: String = "localhost:8980"
  val logger = LogManager.getLogger(classOf[FLSpec])
  Configurator.setLevel("org", Level.ERROR)
  Configurator.setLevel("io", Level.ERROR)
  Configurator.setLevel("com.intel.analytics.bigdl.dllib", Level.ERROR)
  Configurator.setLevel("com.intel.analytics.bigdl.ppml", Level.INFO)

  before {
    // try only next 10 ports, if failed, it may well be
    // that server holds the port and fails to release
    FLContext.resetFLContext()
    port = PortUtils.findNextPortAvailable(port, port + 10)
    target = "localhost:" + port
    logger.info(s"Running test on port: $port, target: $target")

  }
}
