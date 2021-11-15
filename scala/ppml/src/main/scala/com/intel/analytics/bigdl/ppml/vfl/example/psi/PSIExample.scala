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

import com.intel.analytics.bigdl.ppml.vfl.algorithm.PSI


object PSIExample {
  private val logger = LoggerFactory.getLogger(getClass)

  @throws[Exception]
  def main(args: Array[String]): Unit = {
    var max_wait = 20
    // Example code for flClient
    val idSize = 11
    // Quick lookup for the plaintext of hashed ids
    val data = TestUtils.genRandomHashSet(idSize)
    val hashedIds = new util.HashMap[String, String]
    val ids = new util.ArrayList[String](data.keySet)
    // Create a communication channel to the server,  known as a Channel. Channels are thread-safe
    // and reusable. It is common to create channels at the beginning of your application and reuse
    // them until the application shuts down.
    val pSI = new PSI()
    try {
      // Get salt from Server
      val salt = pSI.getSalt()

      logger.debug("Client get Slat=" + salt)
      // Hash(IDs, salt) into hashed IDs
      val hashedIdArray = TestUtils.parallelToSHAHexString(ids, salt)
      for (i <- 0 until ids.size) {
        hashedIds.put(hashedIdArray.get(i), ids.get(i))
      }
      logger.debug("HashedIDs Size = " + hashedIds.size)
      pSI.uploadSet(hashedIdArray)
      var intersection = null
      while ( {
        max_wait > 0
      }) {
        val intersection = pSI.downloadIntersection
        if (intersection == null) {
          logger.info("Wait 1000ms")
          Thread.sleep(1000)
        }
        else {
          logger.info("Intersection successful. Intersection's size is " + intersection.size + ".")

        }
        max_wait -= 1
      }
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


