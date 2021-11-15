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

package com.intel.analytics.bigdl.ppml.vfl.algorithm

import java.util

import com.intel.analytics.bigdl.ppml.psi.test.TestUtils
import com.intel.analytics.bigdl.ppml.vfl.utils.FLClientClosable
import org.apache.log4j.Logger

import scala.collection.JavaConverters._

class PSI() extends FLClientClosable {
  val logger = Logger.getLogger(getClass)
  private var hashedKeyPairs: Map[String, String] = null
  def getHashedKeyPairs() = {
    hashedKeyPairs
  }
  def uploadKeys(keys: Array[String]) = {
    val salt = getSalt
    logger.debug("Client get Salt=" + salt)
    val hashedKeys = TestUtils.parallelToSHAHexString(keys, salt)
    hashedKeyPairs = hashedKeys.zip(keys).toMap
    // Hash(IDs, salt) into hashed IDs
    logger.debug("HashedIDs Size = " + hashedKeys.size)
    uploadSet(hashedKeys.toList.asJava)

  }

  def getSalt(): String = flClient.psiStub.getSalt()
  def getSalt(name: String, clientNum: Int, secureCode: String): String =
    flClient.psiStub.getSalt(name, clientNum, secureCode)

  def uploadSet(hashedIdArray: util.List[String]): Unit = {
    flClient.psiStub.uploadSet(hashedIdArray)
  }

  def downloadIntersection(): util.List[String] = flClient.psiStub.downloadIntersection


}
