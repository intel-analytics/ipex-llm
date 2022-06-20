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

package com.intel.analytics.bigdl.ppml.utils

import org.slf4j.LoggerFactory

/**
 * Log interface.
 */
trait Supportive {
  /**
   * Output time and log.
   * @param name name of log.
   * @param f log content.
   * @tparam T template type.
   * @return a template type result.
   */
  def timing[T](name: String)(f: => T): T = {
    val begin = System.currentTimeMillis
    val result = f
    val end = System.currentTimeMillis
    val cost = (end - begin)
    Supportive.logger.info(s"$name time elapsed $cost ms.")
    result
  }
}

object Supportive {
  val logger = LoggerFactory.getLogger(getClass)
}
