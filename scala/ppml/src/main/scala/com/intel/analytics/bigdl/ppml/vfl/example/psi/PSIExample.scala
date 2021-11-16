/*
 * Copyright 2021 The BigDL Authors.
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

package com.intel.analytics.bigdl.ppml.vfl.example.psi

import org.slf4j.LoggerFactory
import java.util

import com.intel.analytics.bigdl.ppml.psi.HashingUtils
import com.intel.analytics.bigdl.ppml.vfl.{PSI, VflContext}


object PSIExample {
  private val logger = LoggerFactory.getLogger(getClass)

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    var max_wait = 20
    // Example code for flClient
    val idSize = 11
    // Quick lookup for the plaintext of hashed ids
    val data = HashingUtils.genRandomHashSet(idSize)
    val hashedIds = new util.HashMap[String, String]
    val ids = new util.ArrayList[String](data.keySet)
    // Create a communication channel to the server,  known as a Channel. Channels are thread-safe
    // and reusable. It is common to create channels at the beginning of your application and reuse
    // them until the application shuts down.
    VflContext.initContext()
    val pSI = new PSI()
    try {
      // Get salt from Server
      val salt = pSI.getSalt()

      logger.info("Client get Slat=" + salt)
      pSI.uploadSet(ids)
      logger.info("Client uploaded set" + ids.toString)
      val intersection = pSI.downloadIntersection()
      logger.info("Client get intersection=" + intersection.toString)
    } catch {
      case e: Exception =>
        e.printStackTrace()
    } finally {
      // ManagedChannels use resources like threads and TCP connections. To prevent leaking these
      // resources the channel should be shut down when it will no longer be used. If it may be used
      // again leave it running.
      pSI.close()
    }
  }
}


