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

import com.intel.analytics.bigdl.dllib.utils.Log4Error
import com.intel.analytics.bigdl.ppml.fl.example.DebugLogger
import org.scalatest.{BeforeAndAfter, FlatSpec, Matchers}

class FLServerSpec extends FLSpec {
  "start server from config" should "work" in {
    val flServer = new FLServer(Array("-c",
      getClass.getClassLoader.getResource("ppml-conf-2-party.yaml").getPath))
    Log4Error.invalidOperationError(flServer.getPort == 8980,
      "reading ppml-conf.yaml got wrong config")

    val flClient = new FLClient(Array("-c",
      getClass.getClassLoader.getResource("ppml-conf-2-party.yaml").getPath))
    Log4Error.invalidOperationError(flClient.getTarget == "localhost:8980",
      "reading ppml-conf.yaml got wrong config")
  }

}
