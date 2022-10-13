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
package com.intel.analytics.bigdl.ppml.fl.example.ckks

import com.intel.analytics.bigdl.ckks.CKKS
import com.intel.analytics.bigdl.ppml.fl.FLServer

object StartServer {
  def main(args: Array[String]): Unit = {
    val flServer = new FLServer()

    flServer.setClientNum(2)
    if (args.length > 0) {
      val secretsPath = args(0)
      flServer.setCkksAggregator(
        CKKS.loadSecret(secretsPath))
    }
    flServer.build()
    flServer.start()
    flServer.blockUntilShutdown()
  }
}
