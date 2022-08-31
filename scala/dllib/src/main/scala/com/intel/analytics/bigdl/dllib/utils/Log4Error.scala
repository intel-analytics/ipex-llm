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

package com.intel.analytics.bigdl.dllib.utils

import org.apache.logging.log4j.{LogManager, Logger}

object Log4Error {
  val logger: Logger = LogManager.getLogger(getClass)

  def invalidInputError(condition: Boolean, errmsg: String, fixmsg: String = null): Unit = {
    if (!condition) {
      outputUserMessage(errmsg, fixmsg)
      // scalastyle:off
      throw new IllegalArgumentException(errmsg)
      // scalastyle:on
    }
  }

  def invalidOperationError(condition: Boolean, errmsg: String, fixmsg: String = null,
                            cause: Throwable = null): Unit = {
    if (!condition) {
      outputUserMessage(errmsg, fixmsg)
      // scalastyle:off
      throw new InvalidOperationException(errmsg, cause)
      // scalastyle:on
    }
  }

  def outputUserMessage(errmsg: String = null, fixmsg: String = null): Unit = {
    if (errmsg != null) {
      logger.info(s"\n\n****************************Usage Error************************\n"
        + errmsg)
    }
    if (fixmsg != null) {
      logger.info(s"\n\n****************************How to fix*************************\n"
        + fixmsg)
    }
    logger.info(s"\n\n****************************Call Stack*************************")
  }

  def unKnowExceptionError(condition: Boolean, errmsg: String = null,
    fixmsg: String = null, cause: Throwable = null): Unit = {
    if (!condition) {
      outputUserMessage(errmsg, fixmsg)
      // scalastyle:off
      throw new UnKnownException(errmsg, cause)
      // scalastyle:on
    }
  }
}

class InvalidOperationException(message: String, cause: Throwable = null)
  extends Exception(message, cause) {

  def this(message: String) = this(message, null)
}

class UnKnownException(message: String, cause: Throwable = null)
  extends Exception(message, cause) {

  def this(message: String) = this(message, null)
}
